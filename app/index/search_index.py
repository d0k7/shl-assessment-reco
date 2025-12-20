from __future__ import annotations

import json
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from rank_bm25 import BM25Okapi

from app.schemas.catalog import CatalogItem
from app.utils.url_normalize import canonicalize_url


INDEX_DIR = Path("data/index")
META_PATH = INDEX_DIR / "catalog_meta.jsonl"
BM25_PATH = INDEX_DIR / "catalog_bm25.pkl"


_WORD_RE = re.compile(r"[A-Za-z0-9_]+")


def _tokenize(text: str) -> List[str]:
    text = (text or "").lower()
    return _WORD_RE.findall(text)


def item_to_text(it: CatalogItem) -> str:
    parts: List[str] = []
    if it.name:
        parts.append(it.name)
    if it.description:
        parts.append(it.description)
    if it.test_type:
        parts.append(" ".join(it.test_type))
    if it.job_levels:
        parts.append(" ".join(it.job_levels))
    return "\n".join(parts)


@dataclass
class Artifacts:
    bm25: BM25Okapi
    tokens: List[List[str]]
    items: List[CatalogItem]


def save_bm25(bm25: BM25Okapi, tokens: List[List[str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump({"bm25": bm25, "tokens": tokens}, f)


def load_bm25(path: Path) -> Tuple[BM25Okapi, List[List[str]]]:
    with open(path, "rb") as f:
        obj = pickle.load(f)
    return obj["bm25"], obj["tokens"]


def load_metadata_jsonl(path: Path) -> List[CatalogItem]:
    items: List[CatalogItem] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            obj["url"] = canonicalize_url(str(obj.get("url", "")))
            items.append(CatalogItem(**obj))
    return items


def load_artifacts() -> Artifacts:
    if not META_PATH.exists():
        raise FileNotFoundError(f"Missing metadata: {META_PATH}")
    if not BM25_PATH.exists():
        raise FileNotFoundError(f"Missing BM25 index: {BM25_PATH}")

    items = load_metadata_jsonl(META_PATH)
    bm25, tokens = load_bm25(BM25_PATH)
    return Artifacts(bm25=bm25, tokens=tokens, items=items)


def recommend_from_query(query: str, top_k: int, artifacts: Artifacts) -> List[Dict]:
    q_tokens = _tokenize(query)
    if not q_tokens:
        return []

    scores = artifacts.bm25.get_scores(q_tokens)
    scores = np.asarray(scores, dtype=np.float32)

    top_k = max(1, int(top_k))
    idxs = np.argsort(-scores)[:top_k]

    out: List[Dict] = []
    for i in idxs:
        it = artifacts.items[int(i)]
        out.append(
            {
                "name": it.name,
                "url": it.url,
                "description": it.description or "",
                "test_type": it.test_type or [],
                "score": float(scores[int(i)]),
            }
        )
    return out
