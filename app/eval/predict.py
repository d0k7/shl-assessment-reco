from __future__ import annotations

import argparse
from typing import List

import pandas as pd
from sentence_transformers import SentenceTransformer

from app.index.search_index import canonicalize_url, load_faiss_index, load_metadata_jsonl, recommend


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--xlsx", required=True, help="Path to Gen_AI Dataset.xlsx")
    parser.add_argument("--out", default="predictions.csv", help="Output CSV path")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--index", default="data/index/catalog.faiss")
    parser.add_argument("--meta", default="data/index/catalog_meta.jsonl")
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    args = parser.parse_args()

    df = pd.read_excel(args.xlsx, sheet_name="Test-Set")
    if "Query" not in df.columns:
        raise ValueError("Test-Set sheet must have column: Query")

    model = SentenceTransformer(args.model)
    index = load_faiss_index(args.index)
    items = load_metadata_jsonl(args.meta)

    rows: List[dict] = []
    for q in df["Query"].astype(str).tolist():
        q = q.strip()
        recs = recommend(q, model, index, items, top_k=args.k)
        urls = [canonicalize_url(r["url"]) for r in recs]

        # Required submission format: multiple rows per query
        for u in urls:
            rows.append({"Query": q, "Assessment_url": u})

    out_df = pd.DataFrame(rows, columns=["Query", "Assessment_url"])
    out_df.to_csv(args.out, index=False)
    print(f"✅ Wrote {len(out_df)} rows to {args.out}")


if __name__ == "__main__":
    main()
