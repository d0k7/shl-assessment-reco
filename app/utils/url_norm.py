from __future__ import annotations

import re
from urllib.parse import urlparse

_SLUG_NUM_SUFFIX = re.compile(r"-(\d{3,})$")  # e.g. manager-8-0-jfa-4310


def normalize_url_for_match(url: str) -> str:
    """
    Normalizes URLs so evaluation doesn't fail due to:
      - /solutions/products vs /products
      - trailing slash
      - legacy numeric suffix at end of slug (optional)
    This is for *matching* only.
    """
    u = (url or "").strip()
    if not u:
        return ""

    p = urlparse(u)
    path = p.path

    # normalize /solutions prefix away
    path = path.replace("/solutions/products/product-catalog/view/", "/products/product-catalog/view/")

    # strip trailing slash
    if path.endswith("/"):
        path = path[:-1]

    # remove legacy numeric suffix from last slug token
    parts = path.split("/")
    if parts:
        slug = parts[-1]
        slug2 = _SLUG_NUM_SUFFIX.sub("", slug)
        parts[-1] = slug2
        path = "/".join(parts)

    return f"{p.scheme or 'https'}://{p.netloc or 'www.shl.com'}{path}".lower()


def to_submission_url(url: str) -> str:
    """
    Converts to a stable 'catalog-like' URL format.
    We keep the slug, but prefer the /solutions/products/... prefix.
    """
    u = (url or "").strip()
    if not u:
        return u

    p = urlparse(u)
    path = p.path

    # ensure we are under /solutions/products/product-catalog/view/
    path = path.replace("/products/product-catalog/view/", "/solutions/products/product-catalog/view/")
    if not path.startswith("/solutions/products/product-catalog/view/"):
        # best effort: keep path unchanged if it's unexpected
        pass

    # remove trailing slash for consistency with your current outputs
    if path.endswith("/"):
        path = path[:-1]

    host = p.netloc or "www.shl.com"
    scheme = p.scheme or "https"
    return f"{scheme}://{host}{path}"
