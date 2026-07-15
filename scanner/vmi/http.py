"""Shared HTTP helpers with retries, per-domain rate limiting and on-disk caching."""
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

# Per-domain throttle state: each domain gets its own lock + last-request
# timestamp + minimum interval, since different sites have wildly different
# rate-limit tolerances (macrotrends 429s aggressively; stockanalysis and
# finviz and wikipedia are comfortable much faster).
_DOMAIN_MIN_INTERVAL = {
    "macrotrends": 8.0,     # empirically the safe floor to avoid 429s
    "stockanalysis": 0.35,
    "finviz": 0.5,
    "wikipedia": 0.5,
    "default": 0.5,
}
_domain_locks = {d: threading.Lock() for d in _DOMAIN_MIN_INTERVAL}
_domain_last_ts = {d: 0.0 for d in _DOMAIN_MIN_INTERVAL}


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


def _domain_key(url: str, domain_hint: str = None) -> str:
    if domain_hint:
        return domain_hint if domain_hint in _DOMAIN_MIN_INTERVAL else "default"
    for d in _DOMAIN_MIN_INTERVAL:
        if d != "default" and d in url:
            return d
    return "default"


def _throttle(domain: str):
    lock = _domain_locks.setdefault(domain, threading.Lock())
    min_interval = _DOMAIN_MIN_INTERVAL.get(domain, _DOMAIN_MIN_INTERVAL["default"])
    with lock:
        now = time.time()
        last = _domain_last_ts.get(domain, 0.0)
        wait = last + min_interval - now
        if wait > 0:
            time.sleep(wait)
        _domain_last_ts[domain] = time.time()


def _cache_path(url: str) -> str:
    h = hashlib.sha256(url.encode()).hexdigest()[:24]
    return os.path.join(CACHE_DIR, h + ".html")


def get(url: str, use_cache: bool = True, cache_max_age: float = 86400 * 3,
        retries: int = 3, timeout: int = 30, domain_hint: str = None) -> str:
    """GET a URL with per-domain throttling, retries and optional disk cache.

    domain_hint: explicit key into _DOMAIN_MIN_INTERVAL (e.g. "macrotrends")
    to force a throttle profile regardless of what's in the URL. Falls back
    to substring-matching the URL against known domain keys, then "default".
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    cp = _cache_path(url)
    if use_cache and os.path.exists(cp):
        if time.time() - os.path.getmtime(cp) < cache_max_age:
            with open(cp, "r", encoding="utf-8") as f:
                return f.read()

    domain = _domain_key(url, domain_hint)
    last_err = None
    for attempt in range(retries):
        _throttle(domain)
        try:
            r = _session().get(url, timeout=timeout, allow_redirects=True)
            if r.status_code == 200 and len(r.text) > 500:
                with open(cp, "w", encoding="utf-8") as f:
                    f.write(r.text)
                return r.text
            if r.status_code == 404:
                raise LookupError(f"404 for {url}")
            if r.status_code == 429:
                # Back off harder for rate-limited domains before retrying.
                time.sleep(15 * (attempt + 1))
                last_err = RuntimeError(f"HTTP 429 for {url}")
                continue
            last_err = RuntimeError(f"HTTP {r.status_code} for {url}")
        except LookupError:
            raise
        except Exception as e:  # noqa: BLE001
            last_err = e
        time.sleep(1.5 * (attempt + 1) + random.random())
    raise last_err if last_err else RuntimeError(f"failed: {url}")
