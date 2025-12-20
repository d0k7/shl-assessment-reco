from __future__ import annotations

import re
from typing import List


_SLUG_RE = re.compile(r"/view/([^/?#]+)", re.IGNORECASE)


def slug_from_url(url: str) -> str:
    """
    Extract the stable SHL product slug from either:
      https://www.shl.com/products/product-catalog/view/<slug>/
      https://www.shl.com/solutions/products/product-catalog/view/<slug>/
    """
    m = _SLUG_RE.search(url)
    if not m:
        # fallback: last segment
        parts = [p for p in url.split("/") if p]
        return parts[-1].split("?")[0].split("#")[0]
    return m.group(1).strip()


def canonicalize_url(url: str) -> str:
    """
    Force a single canonical form that matches the dataset more often.
    """
    slug = slug_from_url(url)
    return f"https://www.shl.com/solutions/products/product-catalog/view/{slug}"


_TEST_TYPE_LETTERS = {"A", "B", "C", "D", "E", "K", "P", "S"}


def extract_test_types(text: str) -> List[str]:
    """
    Extract test type letters (A/B/C/D/E/K/P/S) from messy scraped description.
    """
    if not text:
        return []
    # common pattern: "Test Type: B K P"
    m = re.search(r"Test\s*Type\s*:\s*([A-Z\s]+)", text, re.IGNORECASE)
    if m:
        letters = [x.strip().upper() for x in m.group(1).split() if x.strip()]
        return [x for x in letters if x in _TEST_TYPE_LETTERS]

    # fallback: find any isolated type letters near "Test Type"
    window = text.upper()
    if "TEST TYPE" in window:
        nearby = window[window.find("TEST TYPE") : window.find("TEST TYPE") + 80]
        letters = re.findall(r"\b([A|B|C|D|E|K|P|S])\b", nearby)
        return [x for x in letters if x in _TEST_TYPE_LETTERS]

    return []


def extract_duration_minutes(text: str) -> int:
    """
    Parse things like:
      "Approximate Completion Time in minutes = 42"
      "minutes = max 30"
      "Untimed"
    Returns 0 if unknown.
    """
    if not text:
        return 0
    t = text.lower()
    if "untimed" in t:
        return 0

    # minutes = 42
    m = re.search(r"minutes\s*=\s*(\d+)", t)
    if m:
        return int(m.group(1))

    # minutes = max 30
    m = re.search(r"minutes\s*=\s*max\s*(\d+)", t)
    if m:
        return int(m.group(1))

    return 0


def extract_remote_support(text: str) -> str:
    """
    If the description contains "Remote Testing", mark Yes else No.
    """
    if not text:
        return "No"
    return "Yes" if "remote testing" in text.lower() else "No"


def normalize_query(q: str) -> str:
    """
    Trim and collapse whitespace.
    """
    q = (q or "").strip()
    q = re.sub(r"\s+", " ", q)
    return q


def keywordize_long_text(q: str, max_len: int = 400) -> str:
    """
    For very long JDs, embeddings often get noisy.
    This compresses text by:
      - keeping first chunk
      - removing excessive punctuation runs
    """
    q = normalize_query(q)
    if len(q) <= max_len:
        return q
    head = q[:max_len]
    head = re.sub(r"[^\w\s\.\,\-\+/#]", " ", head)
    head = re.sub(r"\s+", " ", head).strip()
    return head
