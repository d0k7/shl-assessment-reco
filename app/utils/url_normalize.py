# app/utils/url_normalize.py
from __future__ import annotations

import re
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit


def canonicalize_url(url: str) -> str:
    """
    Minimal, stable URL canonicalizer:
    - trims whitespace
    - lowercases scheme + hostname
    - removes fragments
    - sorts query params
    - normalizes path (removes duplicate slashes, trims trailing slash except '/')
    """
    if not url:
        return ""

    url = url.strip()
    if not url:
        return ""

    # If user pasted without scheme, don't try to guess too hard; just return trimmed
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+\-.]*://", url):
        return url

    parts = urlsplit(url)
    scheme = (parts.scheme or "").lower()
    netloc = (parts.netloc or "").lower()

    # remove default ports
    if scheme == "http" and netloc.endswith(":80"):
        netloc = netloc[:-3]
    if scheme == "https" and netloc.endswith(":443"):
        netloc = netloc[:-4]

    path = parts.path or "/"
    path = re.sub(r"/{2,}", "/", path)
    if path != "/" and path.endswith("/"):
        path = path[:-1]

    # sort query parameters
    q = parse_qsl(parts.query, keep_blank_values=True)
    q.sort(key=lambda x: (x[0], x[1]))
    query = urlencode(q)

    # drop fragment
    fragment = ""

    return urlunsplit((scheme, netloc, path, query, fragment))
