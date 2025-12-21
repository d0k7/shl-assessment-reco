from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from rank_bm25 import BM25Okapi

CATALOG_PATH = Path("data/catalog.jsonl")


def _tokenize(text: str) -> List[str]:
    text = (text or "").lower()
    return re.findall(r"[a-z0-9]+", text)


def _as_list_str(x: Any) -> List[str]:
    if x is None:
        return []
    if isinstance(x, list):
        return [str(v).strip() for v in x if v is not None and str(v).strip()]
    s = str(x).strip()
    if not s:
        return []
    if "," in s:
        return [p.strip() for p in s.split(",") if p.strip()]
    return [p.strip() for p in s.split() if p.strip()]


def _as_yes_no(x: Any) -> str:
    if x is None:
        return "No"
    if isinstance(x, bool):
        return "Yes" if x else "No"
    s = str(x).strip().lower()
    if s in {"true", "yes", "y", "1"}:
        return "Yes"
    if s in {"false", "no", "n", "0"}:
        return "No"
    return "No"


def _safe_int(x: Any) -> int:
    try:
        if x is None:
            return 0
        return int(x)
    except Exception:
        return 0


def _extract_duration_from_description(desc: str) -> int:
    """
    If duration not provided, try to parse:
    'Approximate Completion Time in minutes = 18'
    """
    if not desc:
        return 0
    m = re.search(r"Completion Time in minutes\s*=\s*(\d+)", desc, re.IGNORECASE)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return 0
    return 0


@dataclass(frozen=True)
class CatalogItem:
    name: str
    url: str
    description: str
    duration: int
    remote_support: str
    adaptive_support: str
    test_type: List[str]


@dataclass
class Artifacts:
    items: List[CatalogItem]
    bm25: BM25Okapi
    corpus_tokens: List[List[str]]
    item_tokens: List[set]


def _item_to_text(it: CatalogItem) -> str:
    # ranking text: name + test_type + description
    parts = [it.name, " ".join(it.test_type or []), it.description]
    return "\n".join([p for p in parts if p]).strip()


# --- Keyword gating/boosting ---
# Add more skills if you want (python, aws, salesforce, etc.)
GATE_KEYWORDS: Dict[str, List[str]] = {
    "java": ["java", "j2ee", "jee", "spring", "hibernate", "jdbc", "ejb"],
}


def _detect_gate(query: str) -> Optional[str]:
    q = set(_tokenize(query))
    for gate, kws in GATE_KEYWORDS.items():
        for kw in kws:
            if kw in q:
                return gate
    return None


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

            description = str(obj.get("description") or obj.get("desc") or "").strip()

            duration = _safe_int(
                obj.get("duration")
                or obj.get("assessment_length")
                or obj.get("assessment_length_minutes")
                or obj.get("length_minutes")
            )
            if duration <= 0:
                duration = _extract_duration_from_description(description)

            remote_support = _as_yes_no(obj.get("remote_support") or obj.get("remote_testing"))
            adaptive_support = _as_yes_no(obj.get("adaptive_support") or obj.get("adaptive"))
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

    item_tokens = [set(toks) for toks in corpus_tokens]

    return Artifacts(items=items, bm25=bm25, corpus_tokens=corpus_tokens, item_tokens=item_tokens)


def _rank_indices(scores: List[float]) -> List[int]:
    # stable deterministic ranking
    return sorted(range(len(scores)), key=lambda i: (scores[i], i), reverse=True)


def recommend_from_query(query: str, top_k: int, artifacts: Artifacts) -> List[Dict[str, Any]]:
    q_tokens = _tokenize(query)
    if not q_tokens:
        return []

    gate = _detect_gate(query)

    scores = artifacts.bm25.get_scores(q_tokens)

    # --- Apply gating/boosting ---
    # If gate detected (e.g., java), we prefer items that contain gate keywords.
    if gate:
        gate_kws = set(GATE_KEYWORDS[gate])
        for i in range(len(scores)):
            itoks = artifacts.item_tokens[i]
            # hard preference: if it contains gate keyword, boost
            if itoks.intersection(gate_kws):
                scores[i] = scores[i] + 50.0  # strong boost so java stays on top
            else:
                # small penalty to push irrelevant stuff down
                scores[i] = scores[i] - 10.0

    ranked_idx = _rank_indices(list(scores))[:top_k]

    out: List[Dict[str, Any]] = []
    for i in ranked_idx:
        it = artifacts.items[i]
        score = float(scores[i])

        out.append(
            {
                "name": it.name,
                "url": it.url,
                "description": it.description,
                "duration": it.duration,
                "remote_support": it.remote_support,
                "adaptive_support": it.adaptive_support,
                "test_type": it.test_type or [],
                "score": round(score, 6),
            }
        )

    return out
