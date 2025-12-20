from __future__ import annotations

import json
from pathlib import Path
from typing import List

from rank_bm25 import BM25Okapi

from app.index.search_index import item_to_text, save_bm25, _tokenize
from app.schemas.catalog import CatalogItem
from app.utils.url_normalize import canonicalize_url


CATALOG_JSONL = Path("data/catalog.jsonl")
OUT_DIR = Path("data/index")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_catalog_jsonl(path: Path) -> List[CatalogItem]:
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


def main() -> None:
    print("Loading catalog data...")
    items = load_catalog_jsonl(CATALOG_JSONL)
    print(f"Loaded items: {len(items)}")

    texts = [item_to_text(it) for it in items]
    tokens = [_tokenize(t) for t in texts]

    print("Building BM25...")
    bm25 = BM25Okapi(tokens)

    meta_path = OUT_DIR / "catalog_meta.jsonl"
    bm25_path = OUT_DIR / "catalog_bm25.pkl"

    print("Writing metadata JSONL...")
    with open(meta_path, "w", encoding="utf-8") as f:
        for it in items:
            f.write(it.model_dump_json())
            f.write("\n")

    print("Saving BM25 artifacts...")
    save_bm25(bm25, tokens, bm25_path)

    print("✅ Done.")
    print(f"- META:  {meta_path}")
    print(f"- BM25:  {bm25_path}")


if __name__ == "__main__":
    main()
