from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from rank_bm25 import BM25Okapi

CATALOG_PATH = Path("data/catalog.jsonl")

_WORD_RE = re.compile(r"[a-z0-9\+#\.]+")

STOPWORDS = {
    "a","an","the","and","or","to","of","in","on","for","with","as","at","by","from",
    "is","are","was","were","be","been","being","this","that","these","those",
    "need","needs","needed","want","wants","looking","require","required",
    "good","great","strong","must","should","who","able","ability",
    "team","teams","stakeholder","stakeholders","external","internal","collaborate","collaborating","collaboration",
}

# Skill intent mapping (expand if you want)
SKILL_SYNONYMS: Dict[str, List[str]] = {
    "java": ["java", "j2ee", "jee", "spring", "hibernate", "struts", "jdk"],
    "python": ["python", "django", "flask", "fastapi"],
    "javascript": ["javascript", "js", "node", "nodejs", "react", "angular", "vue"],
    "sql": ["sql", "postgres", "mysql", "oracle", "sqlite"],
}


def canonicalize_url(url: str) -> str:
    u = (url or "").strip()
    if not u:
        return u
    # normalize trailing slash (prevents duplicates)
    if u.endswith("/"):
        u = u.rstrip("/")
    return u


def _tokenize(text: str) -> List[str]:
    text = (text or "").lower()
    toks = _WORD_RE.findall(text)
    # remove stopwords
    return [t for t in toks if t and t not in STOPWORDS]


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


def _bool_to_yesno(x: Optional[bool]) -> str:
    if x is True:
        return "Yes"
    return "No"


def detect_skills(query: str) -> List[str]:
    q = (query or "").lower()
    found: List[str] = []
    for skill, syns in SKILL_SYNONYMS.items():
        if any(s in q for s in syns):
            found.append(skill)
    return found


def contains_skill(text: str, skill: str) -> bool:
    t = (text or "").lower()
    for s in SKILL_SYNONYMS.get(skill, [skill]):
        if s in t:
            return True
    return False


def boost_score(query_skills: List[str], name: str, doc_text: str) -> float:
    """
    Strong boost if assessment NAME contains the skill, medium if description contains it.
    """
    score = 0.0
    n = (name or "").lower()
    d = (doc_text or "").lower()

    for sk in query_skills:
        if contains_skill(n, sk):
            score += 20.0
        elif contains_skill(d, sk):
            score += 8.0

    # small role-intent bonus
    if "developer" in d or "programmer" in d or "software" in d:
        score += 2.0

    return score


@dataclass(frozen=True)
class CatalogItem:
    name: str
    url: str
    description: str = ""
    duration: Optional[int] = None
    remote_support: Optional[bool] = None
    adaptive_support: Optional[bool] = None
    test_type: List[str] = field(default_factory=list)


@dataclass
class Artifacts:
    items: List[CatalogItem]
    bm25: BM25Okapi
    corpus_tokens: List[List[str]]
    texts: List[str]


def _item_to_text(it: CatalogItem) -> str:
    parts = [
        it.name,
        " ".join(it.test_type or []),
        it.description,
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
            url = canonicalize_url(str(obj.get("url", "")).strip())
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

    return Artifacts(items=items, bm25=bm25, corpus_tokens=corpus_tokens, texts=texts)


def recommend_from_query(query: str, top_k: int, artifacts: Artifacts) -> List[Dict[str, Any]]:
    q_tokens = _tokenize(query)
    if not q_tokens:
        return []

    skills = detect_skills(query)

    scores = artifacts.bm25.get_scores(q_tokens)

    # Candidate filtering: if query implies a skill (e.g., java), prefer items that contain it
    candidate_idxs = list(range(len(scores)))
    if skills:
        filtered = []
        for i in candidate_idxs:
            if any(contains_skill(artifacts.texts[i], sk) for sk in skills):
                filtered.append(i)
        # only apply filter if we still have enough results, else fallback
        if len(filtered) >= max(10, top_k):
            candidate_idxs = filtered

    # Rerank with boosts
    reranked: List[Tuple[int, float]] = []
    for i in candidate_idxs:
        base = float(scores[i])
        extra = boost_score(skills, artifacts.items[i].name, artifacts.texts[i])
        reranked.append((i, base + extra))

    reranked.sort(key=lambda x: x[1], reverse=True)

    # Deduplicate by URL
    out: List[Dict[str, Any]] = []
    seen = set()

    for i, final_score in reranked:
        it = artifacts.items[i]
        if it.url in seen:
            continue
        seen.add(it.url)

        out.append(
            {
                "name": it.name,
                "url": it.url,
                "description": it.description,
                "duration": it.duration or 0,
                "remote_support": _bool_to_yesno(it.remote_support),
                "adaptive_support": _bool_to_yesno(it.adaptive_support),
                "test_type": it.test_type or [],
                "score": round(float(final_score), 6),
            }
        )
        if len(out) >= top_k:
            break

    return out
