"""Shared HTTP helpers with retries, per-domain rate limiting and on-disk caching."""
import hashlib
import os
import random
import subprocess
import threading
import time

import requests

# Domains that block Python's `requests`/urllib3 via TLS fingerprinting
# (JA3) even with correct headers, but accept plain `curl` requests fine
# (verified empirically: wikipedia.org 403s requests, 200s curl with the
# identical User-Agent). Shell out to curl for these instead.
_CURL_FALLBACK_DOMAINS = {"wikipedia"}

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/126.0 Safari/537.36")

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cache")

_session_local = threading.local()

# Per-domain throttle state: each domain gets its own lock + last-request
# timestamp + minimum interval, since different sites have wildly different
# rate-limit tolerances (macrotrends 429s aggressively; stockanalysis and
# finviz and wikipedia are comfortable much faster).
_DOMAIN_MIN_INTERVAL = {
    "macrotrends": 8.0,     # fallback source; aggressively rate-limited
    "sec": 0.15,            # below the SEC fair-access ceiling of 10 req/s
    "yahoo": 0.35,
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


def _curl_get(url: str, timeout: int) -> "tuple[int, str]":
    """Fallback fetch via the `curl` binary for domains that TLS-fingerprint
    block Python's requests/urllib3 (see _CURL_FALLBACK_DOMAINS)."""
    try:
        proc = subprocess.run(
            ["curl", "-s", "-L", "-A", UA, "-w", "\n__HTTP_CODE__%{http_code}",
             "--max-time", str(timeout), url],
            capture_output=True, text=True, timeout=timeout + 10,
            # CRITICAL on Windows: text=True alone decodes with the locale
            # codec (cp1252), which crashes the stdout reader thread on any
            # non-cp1252 byte in a UTF-8 page (seen: 0x8d on Wikipedia) and
            # silently returns stdout=None. Force UTF-8 and never crash on
            # a stray byte.
            encoding="utf-8", errors="replace",
        )
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        raise RuntimeError(f"curl fallback failed for {url}: {e}")
    out = proc.stdout
    if out is None:  # reader thread died (e.g. decode error) — treat as failed
        raise RuntimeError(f"curl fallback returned no output for {url}")
    marker = "__HTTP_CODE__"
    idx = out.rfind(marker)
    if idx == -1:
        return 0, out
    code = int(out[idx + len(marker):].strip() or 0)
    body = out[:idx]
    return code, body


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
    use_curl = domain in _CURL_FALLBACK_DOMAINS
    last_err = None
    for attempt in range(retries):
        _throttle(domain)
        try:
            if use_curl:
                status_code, text = _curl_get(url, timeout)
            else:
                headers = None
                if domain == "sec":
                    # SEC fair-access policy requires a declarative identifying
                    # UA ("Name contact"); browser-style UAs get 403'd.
                    # Override SEC_USER_AGENT with a real contact for
                    # scheduled/production runs.
                    headers = {
                        "User-Agent": os.environ.get(
                            "SEC_USER_AGENT", "VMIScanner scanner@example.com"
                        ),
                        "Accept": "application/json",
                    }
                r = _session().get(url, timeout=timeout, allow_redirects=True,
                                   headers=headers)
                status_code, text = r.status_code, r.text
            if status_code == 200 and len(text) > 500:
                with open(cp, "w", encoding="utf-8") as f:
                    f.write(text)
                return text
            if status_code == 404:
                raise LookupError(f"404 for {url}")
            if status_code == 429:
                # Back off harder for rate-limited domains before retrying.
                time.sleep(15 * (attempt + 1))
                last_err = RuntimeError(f"HTTP 429 for {url}")
                continue
            last_err = RuntimeError(f"HTTP {status_code} for {url}")
        except LookupError:
            raise
        except Exception as e:  # noqa: BLE001
            last_err = e
        time.sleep(1.5 * (attempt + 1) + random.random())
    raise last_err if last_err else RuntimeError(f"failed: {url}")
