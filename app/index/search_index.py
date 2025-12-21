from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from rank_bm25 import BM25Okapi

CATALOG_PATH = Path("data/catalog.jsonl")

_WORD_RE = re.compile(r"[a-z0-9]+")

# Keep this small; we don't want to accidentally remove "java" etc.
_STOPWORDS = {
    "a", "an", "the", "and", "or", "to", "of", "for", "with", "in", "on", "at", "by", "from",
    "who", "is", "are", "be", "as", "this", "that", "these", "those",
    "need", "good", "experience", "skills", "skill", "role", "job", "candidate",
}

# HARD skill gates (if query mentions these, restrict candidates)
_SKILL_GROUPS: Dict[str, List[str]] = {
    "java": ["java", "j2ee", "jee", "spring", "hibernate", "struts", "jdbc", "ejb"],
    "python": ["python", "django", "flask", "fastapi"],
    "javascript": ["javascript", "node", "nodejs", "react", "angular", "vue"],
    "sql": ["sql", "mysql", "postgres", "postgresql", "oracle", "sqlite"],
}


def _tokenize(text: str) -> List[str]:
    t = (text or "").lower()
    toks = _WORD_RE.findall(t)
    return [x for x in toks if x and x not in _STOPWORDS]


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
    # normalize trailing slash so streamlit links don't differ
    if u.endswith("/"):
        u = u[:-1]
    return u


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
    # used for gating quickly
    gate_blob_lower: List[str]


def _item_to_text(it: CatalogItem) -> str:
    # BM25 ranking text; repeat name to weight it a bit
    parts = [
        it.name,
        it.name,
        " ".join(it.test_type or []),
        it.description or "",
    ]
    return "\n".join([p for p in parts if p]).strip()


def _detect_gate_raw(query: str) -> Optional[str]:
    q = (query or "").lower()
    for gate, keys in _SKILL_GROUPS.items():
        # raw substring detection so it never fails due to tokenization/stopwords
        if any(k in q for k in keys):
            return gate
    return None


def _matches_gate(blob_lower: str, gate: str) -> bool:
    keys = _SKILL_GROUPS.get(gate, [])
    return any(k in blob_lower for k in keys)


def load_artifacts() -> Artifacts:
    if not CATALOG_PATH.exists():
        raise FileNotFoundError(f"Missing catalog file: {CATALOG_PATH}")

    items: List[CatalogItem] = []
    texts: List[str] = []
    gate_blob_lower: List[str] = []

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
            txt = _item_to_text(it)
            texts.append(txt)

            # include url too for gating
            gate_blob_lower.append(f"{name} {description} {url}".lower())

    corpus_tokens = [_tokenize(t) for t in texts]
    bm25 = BM25Okapi(corpus_tokens)
    return Artifacts(items=items, bm25=bm25, corpus_tokens=corpus_tokens, gate_blob_lower=gate_blob_lower)


def recommend_from_query(query: str, top_k: int, artifacts: Artifacts) -> List[Dict[str, Any]]:
    q_tokens = _tokenize(query)
    if not q_tokens:
        return []

    gate = _detect_gate_raw(query)

    # ✅ HARD candidate filter
    if gate:
        candidates = [i for i, blob in enumerate(artifacts.gate_blob_lower) if _matches_gate(blob, gate)]
        # only fallback if literally nothing matches
        if not candidates:
            candidates = list(range(len(artifacts.items)))
            gate = None
    else:
        candidates = list(range(len(artifacts.items)))

    # Debug: shows in Render logs so you can confirm the gate is applied
    print(f"[recommend] gate={gate} candidates={len(candidates)} top_k={top_k}")

    scores = artifacts.bm25.get_scores(q_tokens)

    # ✅ Extra boost: if the gate keyword appears in NAME, it should outrank generic matches
    if gate:
        keys = _SKILL_GROUPS[gate]
        for i in candidates:
            name_low = artifacts.items[i].name.lower()
            if any(k in name_low for k in keys):
                scores[i] += 500.0  # big, on purpose

    ranked_idx = sorted(candidates, key=lambda i: scores[i], reverse=True)[:top_k]

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
