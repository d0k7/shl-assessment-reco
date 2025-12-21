from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from rank_bm25 import BM25Okapi

CATALOG_PATH = Path("data/catalog.jsonl")

_WORD_RE = re.compile(r"[a-z0-9]+")

# Cut description at these boilerplate markers (your catalog has them a LOT)
_CUTOFF_MARKERS = [
    "job levels",
    "languages",
    "assessment length",
    "test type",
    "remote testing",
    "accelerate your talent strategy",
]

# Common “role noise” words that appear everywhere in SHL blurbs
_STOPWORDS = {
    "need", "good", "collaborating", "collaboration", "external", "teams", "team",
    "stakeholders", "customer", "customers", "communication", "skills", "role",
    "job", "levels", "languages", "assessment", "length", "approximate",
    "completion", "time", "minutes", "test", "type", "remote", "testing",
    "downloads", "accelerate", "talent", "strategy", "speak", "today",
    "products", "transform", "our", "new"
}


def _tokenize(text: str) -> List[str]:
    t = (text or "").lower()
    toks = _WORD_RE.findall(t)
    # light stopword removal helps a lot for this dataset
    return [x for x in toks if x not in _STOPWORDS]


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


def _canonicalize_url(u: str) -> str:
    u = (u or "").strip()
    # remove trailing slash to avoid duplicates (/view/x and /view/x/)
    if u.endswith("/"):
        u = u[:-1]
    return u


def _clean_description(desc: str) -> str:
    d = (desc or "").strip()
    if not d:
        return ""
    low = d.lower()

    # cut at first boilerplate marker occurrence
    cut_at = None
    for m in _CUTOFF_MARKERS:
        idx = low.find(m)
        if idx != -1:
            cut_at = idx if cut_at is None else min(cut_at, idx)

    if cut_at is not None and cut_at > 0:
        d = d[:cut_at].strip()

    # remove repeated “Description” label noise
    d = re.sub(r"\bdescription\b", "", d, flags=re.IGNORECASE).strip()
    return d


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
    """
    Build the text BM25 ranks over.

    Key trick:
    - repeat name to weight it higher (BM25 has no per-field weights)
    - use cleaned description (remove boilerplate)
    """
    name = it.name.strip()
    desc = _clean_description(it.description)
    types = " ".join(it.test_type or [])
    parts = [
        name,
        name,  # weight name more
        name,  # weight name more
        types,
        desc,
    ]
    return "\n".join([p for p in parts if p]).strip()


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
            url = _canonicalize_url(str(obj.get("url", "")).strip())
            if not name or not url:
                continue

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


def recommend_from_query(query: str, top_k: int, artifacts: Artifacts) -> List[Dict[str, Any]]:
    q_tokens = _tokenize(query)
    if not q_tokens:
        return []

    scores = artifacts.bm25.get_scores(q_tokens)
    qset = set(q_tokens)

    # Hard-skill boosts (tuned for SHL catalog noise)
    SKILL_BOOST: Dict[str, float] = {
        "java": 35.0,
        "j2ee": 25.0,
        "jee": 25.0,
        "spring": 15.0,
        "hibernate": 15.0,
        "struts": 10.0,
        "jdbc": 10.0,
        "ejb": 10.0,
    }

    for i, it in enumerate(artifacts.items):
        hay = f"{it.name} {_clean_description(it.description)}".lower()
        for skill, boost in SKILL_BOOST.items():
            if skill in qset and skill in hay:
                scores[i] += boost

    ranked_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

    out: List[Dict[str, Any]] = []
    for i in ranked_idx:
        it = artifacts.items[i]
        out.append(
            {
                "name": it.name,
                "url": it.url,
                "description": it.description or "",
                "duration": int(it.duration or 0),
                "remote_support": _yes_no(it.remote_support),
                "adaptive_support": _yes_no(it.adaptive_support),
                "test_type": it.test_type or [],
                "score": round(float(scores[i]), 6),
            }
        )

    return out
