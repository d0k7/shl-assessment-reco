from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from rank_bm25 import BM25Okapi

CATALOG_PATH = Path("data/catalog.jsonl")
_WORD_RE = re.compile(r"[a-z0-9]+")

# Stopwords to reduce generic JD noise dominating BM25
_STOPWORDS = {
    "need", "good", "in", "a", "an", "the", "and", "or", "to", "of", "for", "with",
    "who", "is", "are", "be", "as", "on", "at", "by", "from",
    "collaborating", "collaboration", "external", "teams", "team", "stakeholders",
    "stakeholder", "customers", "customer", "communication", "skills", "skill",
    "job", "role", "responsibilities", "requirements"
}

# ✅ Skill-gates: if query contains a gate key, we ONLY consider items matching that group
_SKILL_GROUPS: Dict[str, List[str]] = {
    "java": ["java", "j2ee", "jee", "spring", "hibernate", "struts", "jdbc", "ejb"],
    "python": ["python", "django", "flask", "fastapi"],
    "javascript": ["javascript", "js", "node", "nodejs", "react", "angular", "vue"],
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
    # remove trailing slash for consistency
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
    gate_texts_lower: List[str]  # used for skill-gating

def _item_to_text(it: CatalogItem) -> str:
    # ranking text (repeat name to weight it)
    parts = [
        it.name,
        it.name,
        " ".join(it.test_type or []),
        it.description or "",
    ]
    return "\n".join([p for p in parts if p]).strip()

def _detect_gate(query_tokens: List[str]) -> Optional[str]:
    qset = set(query_tokens)
    for gate, keys in _SKILL_GROUPS.items():
        if any(k in qset for k in keys):
            return gate
    return None

def _matches_gate(text_lower: str, gate: str) -> bool:
    keys = _SKILL_GROUPS.get(gate, [])
    return any(k in text_lower for k in keys)

def load_artifacts() -> Artifacts:
    if not CATALOG_PATH.exists():
        raise FileNotFoundError(f"Missing catalog file: {CATALOG_PATH}")

    items: List[CatalogItem] = []
    texts: List[str] = []
    gate_texts_lower: List[str] = []

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

            t = _item_to_text(it)
            texts.append(t)

            # gate text: name + description (lower)
            gate_texts_lower.append(f"{name} {description}".lower())

    corpus_tokens = [_tokenize(t) for t in texts]
    bm25 = BM25Okapi(corpus_tokens)

    return Artifacts(items=items, bm25=bm25, corpus_tokens=corpus_tokens, gate_texts_lower=gate_texts_lower)

def recommend_from_query(query: str, top_k: int, artifacts: Artifacts) -> List[Dict[str, Any]]:
    q_tokens = _tokenize(query)
    if not q_tokens:
        return []

    gate = _detect_gate(q_tokens)

    # ✅ Candidate selection (this is the key fix)
    if gate:
        candidates = [i for i, txt in enumerate(artifacts.gate_texts_lower) if _matches_gate(txt, gate)]
        # if too few matches, fallback to all
        if len(candidates) < min(top_k, 5):
            candidates = list(range(len(artifacts.items)))
            gate = None
    else:
        candidates = list(range(len(artifacts.items)))

    scores = artifacts.bm25.get_scores(q_tokens)

    # If gated, strongly boost name matches to push Java tests to top
    if gate:
        keys = _SKILL_GROUPS[gate]
        for i in candidates:
            name_low = artifacts.items[i].name.lower()
            if any(k in name_low for k in keys):
                scores[i] += 100.0

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
