from __future__ import annotations

import json
from pathlib import Path
from typing import List
from urllib.parse import urlparse, urlunparse

import faiss  # type: ignore
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from app.index.bm25 import BM25Index, save_bm25
from app.schemas.catalog import CatalogItem


CATALOG_JSONL = Path("data/catalog.jsonl")
OUT_DIR = Path("data/index")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def canonicalize_url(url: str) -> str:
    """
    Deterministic URL canonicalization:
    - force https
    - drop query + fragment
    - normalize host (strip www.)
    - remove trailing slash (except root)
    """
    url = (url or "").strip()
    if not url:
        return url

    if "://" not in url:
        url = "https://" + url.lstrip("/")

    p = urlparse(url)
    scheme = "https"
    netloc = (p.netloc or "").lower()
    if netloc.startswith("www."):
        netloc = netloc[4:]

    path = p.path or ""
    if len(path) > 1 and path.endswith("/"):
        path = path[:-1]

    return urlunparse((scheme, netloc, path, "", "", ""))


def load_catalog_jsonl(path: str) -> List[CatalogItem]:
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


def item_to_text(it: CatalogItem) -> str:
    """
    Focused text for embeddings/BM25:
    include name + test_type + description.
    """
    parts = [
        it.name,
        " ".join(it.test_type or []),
        it.description or "",
    ]
    return "\n".join([p for p in parts if p]).strip()


def main() -> None:
    print("Loading catalog data...")
    items = load_catalog_jsonl(str(CATALOG_JSONL))
    print(f"Loaded items: {len(items)}")

    texts = [item_to_text(it) for it in items]

    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    print(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)

    print("Encoding embeddings...")
    emb_list = []
    for i in tqdm(range(0, len(texts), 32), total=(len(texts) + 31) // 32):
        batch = texts[i : i + 32]
        embs = model.encode(batch, convert_to_numpy=True, normalize_embeddings=True)
        emb_list.append(embs.astype(np.float32, copy=False))
    X = np.vstack(emb_list)
    d = X.shape[1]

    print("Building FAISS index (cosine / inner product on normalized vectors)...")
    index = faiss.IndexFlatIP(d)
    index.add(X)

    meta_path = OUT_DIR / "catalog_meta.jsonl"
    faiss_path = OUT_DIR / "catalog.faiss"
    bm25_path = OUT_DIR / "catalog_bm25.pkl"
    info_path = OUT_DIR / "index_info.json"

    print("Writing metadata JSONL...")
    with open(meta_path, "w", encoding="utf-8") as f:
        for it in items:
            f.write(it.model_dump_json())
            f.write("\n")

    print("Building BM25...")
    bm25 = BM25Index.build(texts)
    save_bm25(bm25, str(bm25_path))

    print("Saving FAISS index...")
    faiss.write_index(index, str(faiss_path))

    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "count": len(items),
                "model": model_name,
                "faiss": str(faiss_path),
                "meta": str(meta_path),
                "bm25": str(bm25_path),
            },
            f,
            indent=2,
        )

    print("✅ Done.")
    print(f"- FAISS: {faiss_path}")
    print(f"- META:  {meta_path}")
    print(f"- BM25:  {bm25_path}")
    print(f"- INFO:  {info_path}")


if __name__ == "__main__":
    main()
