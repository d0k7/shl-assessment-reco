from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

from rank_bm25 import BM25Okapi

# Uses the metadata you already committed:
# data/index/catalog_meta.jsonl
DEFAULT_META = Path("data/index/catalog_meta.jsonl")

_WORD_RE = re.compile(r"[A-Za-z0-9]+")


def _tokenize(text: str) -> List[str]:
    # Simple tokenizer: lowercase alphanumerics
    return [m.group(0).lower() for m in _WORD_RE.finditer(text or "")]


def _item_to_text(obj: Dict[str, Any]) -> str:
    # Robust: works even if fields differ slightly
    name = str(obj.get("name") or "")
    desc = str(obj.get("description") or "")
    test_type = obj.get("test_type") or obj.get("testType") or []
    if isinstance(test_type, list):
        test_type_s = " ".join([str(x) for x in test_type])
    else:
        test_type_s = str(test_type)

    # Keep it short-ish; BM25 doesn’t need tons of boilerplate
    return "\n".join([x for x in [name, test_type_s, desc] if x]).strip()


@dataclass
class CatalogRecord:
    name: str
    url: str
    description: str


@dataclass
class BM25Artifacts:
    bm25: BM25Okapi
    records: List[CatalogRecord]


def load_bm25_artifacts(meta_path: Path = DEFAULT_META) -> BM25Artifacts:
    records: List[CatalogRecord] = []
    corpus_tokens: List[List[str]] = []

    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)

            url = str(obj.get("url") or "")
            name = str(obj.get("name") or "")
            description = str(obj.get("description") or "")

            text = _item_to_text(obj)
            tokens = _tokenize(text)

            # Only keep items with a usable URL
            if url and tokens:
                records.append(CatalogRecord(name=name, url=url, description=description))
                corpus_tokens.append(tokens)

    if not records:
        raise RuntimeError(f"No records loaded from meta file: {meta_path}")

    bm25 = BM25Okapi(corpus_tokens)
    return BM25Artifacts(bm25=bm25, records=records)


def recommend_bm25(query: str, artifacts: BM25Artifacts, top_k: int = 10) -> List[Dict[str, Any]]:
    q_tokens = _tokenize(query)
    if not q_tokens:
        return []

    scores = artifacts.bm25.get_scores(q_tokens)
    # Get top_k indices by score (descending)
    idxs = sorted(range(len(scores)), key=lambda i: float(scores[i]), reverse=True)[:top_k]

    out: List[Dict[str, Any]] = []
    for i in idxs:
        rec = artifacts.records[i]
        out.append(
            {
                "name": rec.name,
                "url": rec.url,
                "description": rec.description,
                "score": float(scores[i]),
            }
        )
    return out
