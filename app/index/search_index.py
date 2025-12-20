from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from app.index.bm25 import BM25Store
from app.utils.url_canonical import canonicalize_url

CATALOG_JSONL = Path("data/catalog.jsonl")


@dataclass
class CatalogDoc:
    name: str
    url: str
    description: str
    raw: Dict[str, Any]


@dataclass
class Artifacts:
    docs: List[CatalogDoc]
    bm25: BM25Store


def _safe_str(x: Any) -> str:
    return "" if x is None else str(x)


def _doc_title_text(d: CatalogDoc) -> str:
    return d.name


def _doc_body_text(d: CatalogDoc) -> str:
    raw = d.raw
    parts: List[str] = []
    parts.append(d.description)

    # Optional structured fields (only if present in JSONL)
    for key in ("test_type", "job_levels", "languages", "assessment_length", "remote_testing"):
        v = raw.get(key)
        if isinstance(v, list):
            parts.append(" ".join(map(_safe_str, v)))
        elif isinstance(v, str):
            parts.append(v)

    return "\n".join([p for p in parts if p]).strip()


def load_catalog_jsonl(path: Path = CATALOG_JSONL) -> List[CatalogDoc]:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Commit data/catalog.jsonl to the repo for Render BM25 mode."
        )

    docs: List[CatalogDoc] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)

            name = _safe_str(obj.get("name"))
            url = canonicalize_url(_safe_str(obj.get("url")))
            desc = _safe_str(obj.get("description"))

            docs.append(CatalogDoc(name=name, url=url, description=desc, raw=obj))

    if not docs:
        raise RuntimeError(f"{path} loaded but contains 0 items.")

    return docs


def load_artifacts() -> Artifacts:
    docs = load_catalog_jsonl()
    titles = [_doc_title_text(d) for d in docs]
    bodies = [_doc_body_text(d) for d in docs]
    bm25 = BM25Store.build(titles, bodies)
    return Artifacts(docs=docs, bm25=bm25)


def recommend_from_query(query: str, top_k: int, artifacts: Artifacts) -> List[Dict[str, Any]]:
    top_k = max(1, min(int(top_k), 25))

    titles_raw = [d.name for d in artifacts.docs]
    hits = artifacts.bm25.topk(query, top_k, titles_raw=titles_raw)

    out: List[Dict[str, Any]] = []
    for idx, score in hits:
        d = artifacts.docs[idx]
        rec = dict(d.raw)
        rec["name"] = d.name
        rec["url"] = d.url
        rec["description"] = d.description
        rec["score"] = score
        out.append(rec)

    return out
