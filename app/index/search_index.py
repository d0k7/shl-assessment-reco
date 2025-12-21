from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from rank_bm25 import BM25Okapi

CATALOG_PATH = Path("data/catalog.jsonl")

_WORD_RE = re.compile(r"[a-z0-9]+")

# Cut long SHL boilerplate so ranking uses the real “skill” part of the text
_CUTOFF_MARKERS = [
    "job levels",
    "languages",
    "assessment length",
    "test type",
    "remote testing",
    "accelerate your talent strategy",
]

# Generic words that appear everywhere and ruin BM25 relevance
_STOPWORDS = {
    "need", "good", "collaborating", "collaboration", "external", "teams", "team",
    "stakeholders", "stakeholder", "customer", "customers", "communication",
    "skills", "skill", "role", "job", "levels", "languages",
    "assessment", "length", "approximate", "completion", "time", "minutes",
    "test", "type", "remote", "testing", "downloads",
    "accelerate", "talent", "strategy", "speak", "today", "products", "transform",
    "our", "new", "report", "solution"
}

# ✅ Skill gates: if query contains one of these skill keys, we only return items matching the same group
# (This makes "Java developer" return Java tests, not random admin/manager tests.)
_SKILL_GROUPS: Dict[str, List[str]] = {
    "java": ["java", "j2ee", "jee", "spring", "hibernate", "struts", "jdbc", "ejb"],
    "python": ["python", "django", "flask", "fastapi"],
    "javascript": ["javascript", "js", "node", "nodejs", "react", "angular", "vue"],
    "sql": ["sql", "mysql", "postgres", "postgresql", "oracle", "mongodb"],
    "aws": ["aws", "ec2", "s3", "lambda", "cloudwatch"],
}


def _tokenize(text: str) -> List[str]:
    t = (text or "").lower()
    toks = _WORD_RE.findall(t)
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
    # remove trailing slash to avoid duplicates
    if u.endswith("/"):
        u = u[:-1]
    return u


def _clean_description(desc: str) -> str:
    d = (desc or "").strip()
    if not d:
        return ""
    low = d.lower()

    cut_at = None
    for m in _CUTOFF_MARKERS:
        idx = low.find(m)
        if idx != -1:
            cut_at = idx if cut_at is None else min(cut_at, idx)

    if cut_at is not None and cut_at > 0:
        d = d[:cut_at].strip()

    d = re.sub(r"\bdescription\b", "", d, flags=re.IGNORECASE).strip()
    return d


def _detect_skill_gate(query_tokens: List[str]) -> Optional[str]:
    qset = set(query_tokens)
    for group, keys in _SKILL_GROUPS.items():
        if any(k in qset for k in keys):
            return group
    return None


def _item_matches_skill_group(item_text_lower: str, group: str) -> bool:
    keys = _SKILL_GROUPS.get(group, [])
    return any(k in item_text_lower for k in keys)


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
    item_search_texts_lower: List[str]  # for skill-gating


def _item_to_text_for_ranking(it: CatalogItem) -> str:
    """
    BM25 ranking text.

    Trick: repeat name to weight it more.
    Use cleaned description to avoid SHL boilerplate dominating ranking.
    """
    name = it.name.strip()
    desc = _clean_description(it.description)
    types = " ".join(it.test_type or [])
    parts = [
        name,
        name,
        name,
        types,
        desc,
    ]
    return "\n".join([p for p in parts if p]).strip()


def load_artifacts() -> Artifacts:
    if not CATALOG_PATH.exists():
        raise FileNotFoundError(f"Missing catalog file: {CATALOG_PATH}")

    items: List[CatalogItem] = []
    texts: List[str] = []
    item_search_texts_lower: List[str] = []

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

            ranking_text = _item_to_text_for_ranking(it)
            texts.append(ranking_text)

            # For gating we use a simpler text: name + cleaned desc
            gate_text = f"{name} {_clean_description(description)}".lower()
            item_search_texts_lower.append(gate_text)

    corpus_tokens = [_tokenize(t) for t in texts]
    bm25 = BM25Okapi(corpus_tokens)

    return Artifacts(
        items=items,
        bm25=bm25,
        corpus_tokens=corpus_tokens,
        item_search_texts_lower=item_search_texts_lower,
    )


def recommend_from_query(query: str, top_k: int, artifacts: Artifacts) -> List[Dict[str, Any]]:
    q_tokens = _tokenize(query)
    if not q_tokens:
        return []

    # ✅ 1) Skill gating (the “one fix” that solves your Java issue)
    gate = _detect_skill_gate(q_tokens)

    # Candidate indices
    candidates: List[int]
    if gate:
        candidates = [
            i for i, txt in enumerate(artifacts.item_search_texts_lower)
            if _item_matches_skill_group(txt, gate)
        ]
        # If gate found almost nothing, fallback to full search
        if len(candidates) < min(top_k, 5):
            candidates = list(range(len(artifacts.items)))
            gate = None
    else:
        candidates = list(range(len(artifacts.items)))

    # ✅ 2) Rank with BM25, but only among candidates
    scores = artifacts.bm25.get_scores(q_tokens)

    # Additional boost: if the gate exists, strongly prefer name matches
    if gate:
        keys = _SKILL_GROUPS[gate]
        for i in candidates:
            name_low = artifacts.items[i].name.lower()
            if any(k in name_low for k in keys):
                scores[i] += 50.0  # name match should dominate

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
