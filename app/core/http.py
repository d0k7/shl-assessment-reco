from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FetchConfig:
    timeout_s: int = 60
    min_delay_s: float = 0.4
    max_delay_s: float = 1.0
    max_retries: int = 2           # retries inside urllib3
    backoff_factor: float = 0.7    # backoff inside urllib3


# A couple realistic UA strings (rotate if blocked)
_UA_POOL = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
]


def build_session(user_agent: str, cfg: FetchConfig) -> requests.Session:
    """
    Requests Session with retries for transient failures.
    We keep headers browser-like but NOT over-specified.
    IMPORTANT: don't advertise 'br' (brotli) because requests may not decode it without extra libs.
    """
    session = requests.Session()

    session.headers.update(
        {
            "User-Agent": user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,application/json;q=0.8,*/*;q=0.7",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate",  # no 'br'
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
        }
    )

    retry = Retry(
        total=cfg.max_retries,
        connect=cfg.max_retries,
        read=cfg.max_retries,
        status=cfg.max_retries,
        backoff_factor=cfg.backoff_factor,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET", "HEAD"),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=50, pool_maxsize=50)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def _polite_sleep(cfg: FetchConfig) -> None:
    time.sleep(random.uniform(cfg.min_delay_s, cfg.max_delay_s))


def _prime_session(session: requests.Session, cfg: FetchConfig) -> None:
    """
    Many WAFs require a cookie set from the homepage before allowing deeper paths.
    Best-effort: never raise here.
    """
    try:
        _polite_sleep(cfg)
        session.get("https://www.shl.com/", timeout=cfg.timeout_s, allow_redirects=True)
    except Exception:
        pass


def _request_once(
    session: requests.Session,
    url: str,
    cfg: FetchConfig,
    extra_headers: Optional[Dict[str, str]] = None,
) -> Tuple[int, bytes]:
    headers = dict(session.headers)
    if extra_headers:
        headers.update(extra_headers)

    resp = session.get(url, timeout=cfg.timeout_s, headers=headers, stream=True, allow_redirects=True)
    return resp.status_code, resp.content


def fetch_bytes(
    session: requests.Session,
    url: str,
    cfg: FetchConfig,
    extra_headers: Optional[Dict[str, str]] = None,
) -> bytes:
    """
    Hardened fetch:
    - try normal
    - if 405/403: prime session + retry with Referer
    - if still blocked: rotate UA + retry
    """
    _polite_sleep(cfg)

    status, content = _request_once(session, url, cfg, extra_headers=extra_headers)
    if status < 400:
        return content

    # WAF-style blocks often manifest as 403/405
    if status in (403, 405):
        _prime_session(session, cfg)
        _polite_sleep(cfg)
        status2, content2 = _request_once(
            session,
            url,
            cfg,
            extra_headers={**(extra_headers or {}), "Referer": "https://www.shl.com/"},
        )
        if status2 < 400:
            return content2

        # Rotate UA and try once more
        new_ua = random.choice([ua for ua in _UA_POOL if ua != session.headers.get("User-Agent")]) or _UA_POOL[0]
        session.headers["User-Agent"] = new_ua
        _prime_session(session, cfg)
        _polite_sleep(cfg)
        status3, content3 = _request_once(
            session,
            url,
            cfg,
            extra_headers={**(extra_headers or {}), "Referer": "https://www.shl.com/"},
        )
        if status3 < 400:
            return content3

        logger.warning("HTTP %s for %s (after WAF retries)", status3, url)
        raise requests.HTTPError(f"HTTP {status3} for {url}")

    logger.warning("HTTP %s for %s", status, url)
    raise requests.HTTPError(f"HTTP {status} for {url}")


def fetch_text(
    session: requests.Session,
    url: str,
    cfg: FetchConfig,
    extra_headers: Optional[Dict[str, str]] = None,
) -> str:
    data = fetch_bytes(session, url, cfg, extra_headers=extra_headers)
    try:
        return data.decode("utf-8", errors="replace")
    except Exception:
        return data.decode(errors="replace")
