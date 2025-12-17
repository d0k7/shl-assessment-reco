from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Tuple

from bs4 import BeautifulSoup

from app.core.text import extract_main_text, normalize_space, parse_duration_minutes, to_yes_no

logger = logging.getLogger(__name__)


def _collect_jsonld(soup: BeautifulSoup) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for tag in soup.select('script[type="application/ld+json"]'):
        raw = tag.get_text(strip=True)
        if not raw:
            continue
        try:
            obj = json.loads(raw)
            if isinstance(obj, dict):
                out.append(obj)
            elif isinstance(obj, list):
                out.extend([x for x in obj if isinstance(x, dict)])
        except Exception:
            continue
    return out


def _kv_from_tables(soup: BeautifulSoup) -> Dict[str, str]:
    kv: Dict[str, str] = {}

    for row in soup.select("table tr"):
        th = row.select_one("th")
        td = row.select_one("td")
        if not th or not td:
            continue
        k = normalize_space(th.get_text(" ", strip=True)).lower()
        v = normalize_space(td.get_text(" ", strip=True))
        if k and v:
            kv[k] = v

    for dt in soup.select("dl dt"):
        dd = dt.find_next_sibling("dd")
        if not dd:
            continue
        k = normalize_space(dt.get_text(" ", strip=True)).lower()
        v = normalize_space(dd.get_text(" ", strip=True))
        if k and v:
            kv[k] = v

    return kv


def _find_title(soup: BeautifulSoup) -> str:
    h1 = soup.select_one("h1")
    if h1:
        t = normalize_space(h1.get_text(" ", strip=True))
        if t:
            return t
    title = soup.select_one("title")
    return normalize_space(title.get_text(" ", strip=True) if title else "")


def _extract_test_type_from_text(text: str) -> List[str]:
    m = re.search(r"\bTest\s*Type\s*[:\-]\s*([A-Za-z, /]+)\b", text, re.IGNORECASE)
    if not m:
        return []
    raw = m.group(1)
    parts = re.split(r"[,/]| and ", raw)
    cleaned = [normalize_space(p).upper() for p in parts if normalize_space(p)]
    return cleaned[:10]


def parse_item(html: str, url: str) -> Tuple[str, str, Dict[str, Any]]:
    soup = BeautifulSoup(html, "lxml")
    title = _find_title(soup)
    main_text = extract_main_text(html)
    kv = _kv_from_tables(soup)
    meta: Dict[str, Any] = {"kv": kv, "jsonld": _collect_jsonld(soup)}
    return title, main_text, meta


def map_fields(title: str, main_text: str, meta: Dict[str, Any]) -> Dict[str, Any]:
    kv: Dict[str, str] = meta.get("kv", {}) or {}
    jsonlds: List[Dict[str, Any]] = meta.get("jsonld", []) or []

    description = (main_text or "")[:5000]

    duration = None
    for k in ("duration", "assessment length", "time", "time to complete", "length"):
        if k in kv:
            duration = parse_duration_minutes(kv[k])
            if duration is not None:
                break
    if duration is None:
        duration = parse_duration_minutes(main_text)

    adaptive_support = None
    for k in ("adaptive", "adaptive support", "adaptive testing"):
        if k in kv:
            adaptive_support = to_yes_no(kv[k])
            if adaptive_support:
                break

    remote_support = None
    for k in ("remote", "remote support", "remote testing", "remote proctoring"):
        if k in kv:
            remote_support = to_yes_no(kv[k])
            if remote_support:
                break

    test_type: List[str] = []
    for k in ("test type", "test types", "type"):
        if k in kv:
            parts = re.split(r"[,/]| and ", kv[k])
            test_type = [normalize_space(p).upper() for p in parts if normalize_space(p)]
            break
    if not test_type:
        test_type = _extract_test_type_from_text(main_text)

    # JSON-LD fallback (defensive)
    for obj in jsonlds:
        if not isinstance(obj, dict):
            continue
        if (not title) and isinstance(obj.get("name"), str):
            title = normalize_space(obj["name"])
        if (not description or len(description) < 80) and isinstance(obj.get("description"), str):
            description = normalize_space(obj["description"])[:5000]

    return {
        "name": title,
        "description": description,
        "duration": int(duration or 0),
        "adaptive_support": adaptive_support or "No",
        "remote_support": remote_support or "No",
        "test_type": test_type[:10],
    }
