"""Shared HTTP helpers with retries, rate limiting and on-disk caching."""
import hashlib
import os
import random
import threading
import time

import requests

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/126.0 Safari/537.36")

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cache")

_session_local = threading.local()
_rate_lock = threading.Lock()
_last_request_ts = 0.0

# Minimum seconds between requests (global, across threads)
MIN_INTERVAL = 0.35


def _session() -> requests.Session:
    s = getattr(_session_local, "s", None)
    if s is None:
        s = requests.Session()
        s.headers.update({
            "User-Agent": UA,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        })
        _session_local.s = s
    return s


def _throttle():
    global _last_request_ts
    with _rate_lock:
        now = time.time()
        wait = _last_request_ts + MIN_INTERVAL - now
        if wait > 0:
            time.sleep(wait)
        _last_request_ts = time.time()


def _cache_path(url: str) -> str:
    h = hashlib.sha256(url.encode()).hexdigest()[:24]
    return os.path.join(CACHE_DIR, h + ".html")


def get(url: str, use_cache: bool = True, cache_max_age: float = 86400 * 3,
        retries: int = 3, timeout: int = 30) -> str:
    """GET a URL with throttling, retries and optional disk cache."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    cp = _cache_path(url)
    if use_cache and os.path.exists(cp):
        if time.time() - os.path.getmtime(cp) < cache_max_age:
            with open(cp, "r", encoding="utf-8") as f:
                return f.read()

    last_err = None
    for attempt in range(retries):
        _throttle()
        try:
            r = _session().get(url, timeout=timeout, allow_redirects=True)
            if r.status_code == 200 and len(r.text) > 500:
                with open(cp, "w", encoding="utf-8") as f:
                    f.write(r.text)
                return r.text
            if r.status_code == 404:
                raise LookupError(f"404 for {url}")
            last_err = RuntimeError(f"HTTP {r.status_code} for {url}")
        except LookupError:
            raise
        except Exception as e:  # noqa: BLE001
            last_err = e
        time.sleep(1.5 * (attempt + 1) + random.random())
    raise last_err if last_err else RuntimeError(f"failed: {url}")
