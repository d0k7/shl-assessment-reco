from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, List, Optional, Tuple

import faiss  # type: ignore
import numpy as np
from sentence_transformers import SentenceTransformer

from app.schemas.catalog import CatalogItem
from app.schemas.api import RecommendedAssessment
from app.utils.text import looks_like_url, fetch_url_text

ROOT_DIR = Path(__file__).resolve().parents[2]   # C:\...\shl-reco\shl-reco
DATA_DIR = ROOT_DIR / "data"
INDEX_DIR = DATA_DIR / "index"

DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_FAISS_PATH = INDEX_DIR / "catalog.faiss"
DEFAULT_META_PATH = INDEX_DIR / "catalog_meta.jsonl"


def load_metadata_jsonl(path: Path) -> List[CatalogItem]:
    items: List[CatalogItem] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(CatalogItem(**json.loads(line)))
    return items


def load_faiss_index(path: Path) -> faiss.Index:
    if not path.exists():
        raise FileNotFoundError(f"FAISS index not found: {path}")
    return faiss.read_index(str(path))


def encode_query(model: SentenceTransformer, text: str) -> np.ndarray:
    vec = model.encode([text], convert_to_numpy=True, normalize_embeddings=True)
    return vec.astype(np.float32, copy=False)


@dataclass(frozen=True)
class Artifacts:
    model: SentenceTransformer
    index: faiss.Index
    items: List[CatalogItem]


@lru_cache(maxsize=1)
def load_artifacts(
    model_name: str = DEFAULT_MODEL_NAME,
    faiss_path: Path = DEFAULT_FAISS_PATH,
    meta_path: Path = DEFAULT_META_PATH,
) -> Artifacts:
    model = SentenceTransformer(model_name)
    index = load_faiss_index(faiss_path)
    items = load_metadata_jsonl(meta_path)
    return Artifacts(model=model, index=index, items=items)


def dense_recommend(query_text: str, artifacts: Artifacts, top_k: int) -> List[Tuple[int, float]]:
    q = encode_query(artifacts.model, query_text)
    scores, idxs = artifacts.index.search(q, top_k)

    out: List[Tuple[int, float]] = []
    for score, idx in zip(scores[0].tolist(), idxs[0].tolist()):
        if idx < 0 or idx >= len(artifacts.items):
            continue
        out.append((idx, float(score)))
    return out


def recommend_from_query(query: str, top_k: int, artifacts: Artifacts) -> List[RecommendedAssessment]:
    q = query.strip()

    if looks_like_url(q):
        try:
            fetched = fetch_url_text(q)
            q_text = f"Job Description from URL:\n{fetched}"
        except Exception:
            q_text = q
    else:
        q_text = q

    hits = dense_recommend(q_text, artifacts, top_k=top_k)

    recs: List[RecommendedAssessment] = []
    for idx, _score in hits:
        it = artifacts.items[idx]
        recs.append(
            RecommendedAssessment(
                name=it.name,
                url=it.url,
                description=getattr(it, "description", "") or "",
                duration=int(getattr(it, "duration", 0) or 0),
                remote_support=str(getattr(it, "remote_support", "No") or "No"),
                adaptive_support=str(getattr(it, "adaptive_support", "No") or "No"),
                test_type=list(getattr(it, "test_type", []) or []),
            )
        )
    return recs
