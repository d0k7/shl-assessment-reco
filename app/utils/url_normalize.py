from __future__ import annotations

from urllib.parse import urlparse, urlunparse


def normalize_shl_url(url: str) -> str:
    """
    Normalize SHL catalog URLs so different site variants match:

    - Remove '/solutions' prefix after domain
      (e.g. https://www.shl.com/solutions/products/... -> https://www.shl.com/products/...)
    - Strip trailing slash
    - Keep scheme+netloc+path only (ignore query/fragment)
    """
    if not url:
        return url

    u = url.strip()
    parsed = urlparse(u)

    scheme = parsed.scheme or "https"
    netloc = parsed.netloc or "www.shl.com"

    path = parsed.path or ""
    # remove trailing slash
    if path.endswith("/"):
        path = path[:-1]

    # remove "/solutions" once at start
    if path.startswith("/solutions/"):
        path = path.replace("/solutions/", "/", 1)

    # drop query/fragment
    normalized = urlunparse((scheme, netloc, path, "", "", ""))
    return normalized
