# app/index/bm25_runtime.py
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

from rank_bm25 import BM25Okapi


_WORD_RE = re.compile(r"[A-Za-z0-9]+")


def _repo_root() -> Path:
    # .../app/index/bm25_runtime.py -> repo root is parents[2]
    return Path(__file__).resolve().parents[2]


def canonicalize_url(url: str) -> str:
    """
    Simple, dependency-free canonicalization good enough for this assignment.
    - trims whitespace
    - removes URL fragments
    """
    u = (url or "").strip()
    if "#" in u:
        u = u.split("#", 1)[0]
    return u


def tokenize(text: str) -> List[str]:
    return [m.group(0).lower() for m in _WORD_RE.finditer(text or "")]


def item_to_text(item: Dict[str, Any]) -> str:
    # Weighted-ish text: name + test_type + description
    name = str(item.get("name", "") or "")
    desc = str(item.get("description", "") or "")
    ttype = item.get("test_type", [])
    if not isinstance(ttype, list):
        ttype = [str(ttype)]
    return "\n".join([name, " ".join(map(str, ttype)), desc]).strip()


@dataclass
class BM25Artifacts:
    items: List[Dict[str, Any]]
    bm25: BM25Okapi
    corpus_tokens: List[List[str]]


def load_catalog_jsonl(path: Path) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            obj["url"] = canonicalize_url(str(obj.get("url", "") or ""))
            items.append(obj)
    return items


def build_bm25_from_catalog() -> BM25Artifacts:
    catalog_path = _repo_root() / "data" / "catalog.jsonl"
    if not catalog_path.exists():
        raise FileNotFoundError(f"Missing catalog.jsonl at: {catalog_path}")

    items = load_catalog_jsonl(catalog_path)
    texts = [item_to_text(it) for it in items]
    corpus_tokens = [tokenize(t) for t in texts]
    bm25 = BM25Okapi(corpus_tokens)
    return BM25Artifacts(items=items, bm25=bm25, corpus_tokens=corpus_tokens)


def recommend_bm25(query: str, top_k: int, artifacts: BM25Artifacts) -> List[Dict[str, Any]]:
    q_tokens = tokenize(query)
    scores = artifacts.bm25.get_scores(q_tokens)  # List[float] / np array
    # Get top-k indices
    pairs: List[Tuple[int, float]] = [(i, float(scores[i])) for i in range(len(scores))]
    pairs.sort(key=lambda x: x[1], reverse=True)
    pairs = pairs[: max(1, int(top_k))]

    out: List[Dict[str, Any]] = []
    for idx, score in pairs:
        it = artifacts.items[idx]
        out.append(
            {
                "name": it.get("name"),
                "url": it.get("url"),
                "description": it.get("description"),
                "duration": it.get("duration"),
                "remote_support": it.get("remote_support"),
                "adaptive_support": it.get("adaptive_support"),
                "test_type": it.get("test_type"),
                "score": round(float(score), 6),
            }
        )
    return out
