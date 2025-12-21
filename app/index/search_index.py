from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from rank_bm25 import BM25Okapi

CATALOG_PATH = Path("data/catalog.jsonl")


# -----------------------------
# Text utilities
# -----------------------------
_WORD_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> List[str]:
    text = (text or "").lower()
    return _WORD_RE.findall(text)


def _safe_int(x: Any) -> Optional[int]:
    try:
        if x is None:
            return None
        return int(x)
    except Exception:
        return None


def _as_bool(x: Any) -> Optional[bool]:
    if x is None:
        return None
    if isinstance(x, bool):
        return x
    s = str(x).strip().lower()
    if s in {"true", "yes", "y", "1"}:
        return True
    if s in {"false", "no", "n", "0"}:
        return False
    return None


def _as_list_str(x: Any) -> List[str]:
    if x is None:
        return []
    if isinstance(x, list):
        return [str(v) for v in x if v is not None]

    s = str(x).strip()
    if not s:
        return []
    if "," in s:
        return [p.strip() for p in s.split(",") if p.strip()]
    return [p.strip() for p in s.split() if p.strip()]


def _yes_no(x: Optional[bool]) -> str:
    return "Yes" if x is True else "No"


# -----------------------------
# Data structures
# -----------------------------
@dataclass(frozen=True)
class CatalogItem:
    name: str
    url: str
    description: str = ""
    duration: Optional[int] = None
    remote_support: Optional[bool] = None
    adaptive_support: Optional[bool] = None
    test_type: List[str] = None  # type: ignore[assignment]


@dataclass
class Artifacts:
    items: List[CatalogItem]
    bm25: BM25Okapi
    corpus_tokens: List[List[str]]


def _item_to_text(it: CatalogItem) -> str:
    # This is what BM25 ranks over (keep it compact but informative).
    parts = [
        it.name,
        " ".join(it.test_type or []),
        it.description,
    ]
    return "\n".join([p for p in parts if p]).strip()


# -----------------------------
# Load catalog + build BM25
# -----------------------------
def load_artifacts() -> Artifacts:
    if not CATALOG_PATH.exists():
        raise FileNotFoundError(f"Missing catalog file: {CATALOG_PATH}")

    items: List[CatalogItem] = []
    texts: List[str] = []

    with open(CATALOG_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            obj: Dict[str, Any] = json.loads(line)

            name = str(obj.get("name", "")).strip()
            url = str(obj.get("url", "")).strip()
            if not name or not url:
                continue

            # --- map fields robustly (catalog.jsonl may vary) ---
            description = str(obj.get("description") or obj.get("desc") or "").strip()

            duration = _safe_int(
                obj.get("duration")
                or obj.get("assessment_length")
                or obj.get("assessment_length_minutes")
                or obj.get("length_minutes")
            )

            remote_support = _as_bool(obj.get("remote_support") or obj.get("remote_testing"))
            adaptive_support = _as_bool(obj.get("adaptive_support") or obj.get("adaptive"))

            test_type = _as_list_str(obj.get("test_type") or obj.get("testTypes") or obj.get("type_codes"))

            it = CatalogItem(
                name=name,
                url=url,
                description=description,
                duration=duration,
                remote_support=remote_support,
                adaptive_support=adaptive_support,
                test_type=test_type,
            )

            items.append(it)
            texts.append(_item_to_text(it))

    corpus_tokens = [_tokenize(t) for t in texts]
    bm25 = BM25Okapi(corpus_tokens)

    return Artifacts(items=items, bm25=bm25, corpus_tokens=corpus_tokens)


# -----------------------------
# Recommend
# -----------------------------
def recommend_from_query(query: str, top_k: int, artifacts: Artifacts) -> List[Dict[str, Any]]:
    q_tokens = _tokenize(query)
    if not q_tokens:
        return []

    # Base BM25 scores
    scores = artifacts.bm25.get_scores(q_tokens)
    qset = set(q_tokens)

    # ✅ Hard-skill boosts:
    # BM25 may over-rank generic "collaborate / customer / stakeholders" text.
    # For role queries like "Java developer", we must strongly promote Java items.
    SKILL_BOOST: Dict[str, float] = {
        "java": 20.0,
        "j2ee": 20.0,
        "jee": 20.0,
        "spring": 10.0,
        "hibernate": 10.0,
        "struts": 10.0,
    }

    for i, it in enumerate(artifacts.items):
        text = f"{it.name} {it.description}".lower()
        for skill, boost in SKILL_BOOST.items():
            if skill in qset and skill in text:
                scores[i] += boost

    # Rank desc and slice
    ranked_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

    out: List[Dict[str, Any]] = []
    for i in ranked_idx:
        it = artifacts.items[i]
        score = float(scores[i])

        out.append(
            {
                "name": it.name,
                "url": it.url,
                "description": it.description or "",
                "duration": int(it.duration or 0),
                "remote_support": _yes_no(it.remote_support),
                "adaptive_support": _yes_no(it.adaptive_support),
                "test_type": it.test_type or [],
                "score": round(score, 6),
            }
        )

    return out
