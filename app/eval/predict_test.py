from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss  # type: ignore

from app.index.search_index import load_metadata_jsonl, recommend


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate submission CSV for Test-Set")
    parser.add_argument("--xlsx", default="data/Gen_AI Dataset.xlsx", help="Path to Gen_AI Dataset.xlsx")
    parser.add_argument("--out", default="data/submission.csv", help="Output CSV path")
    parser.add_argument("--top-k", type=int, default=10, help="Number of predictions per query (<=10)")
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--index", default="data/index/catalog.faiss")
    parser.add_argument("--meta", default="data/index/catalog_meta.jsonl")
    args = parser.parse_args()

    if args.top_k < 1 or args.top_k > 10:
        raise ValueError("--top-k must be between 1 and 10")

    xlsx_path = Path(args.xlsx)
    if not xlsx_path.exists():
        raise FileNotFoundError(f"Excel dataset not found: {xlsx_path}")

    print("Loading Test-Set...")
    df = pd.read_excel(str(xlsx_path), sheet_name="Test-Set")
    if "Query" not in df.columns:
        raise ValueError("Test-Set sheet must have column: Query")

    queries: List[str] = [str(q).strip() for q in df["Query"].tolist()]
    queries = [q for q in queries if q]
    print(f"Loaded {len(queries)} test queries")

    print("Loading model + FAISS index + metadata...")
    model = SentenceTransformer(args.model)
    index = faiss.read_index(args.index)
    items = load_metadata_jsonl(args.meta)

    rows = []
    for q in queries:
        recs = recommend(query=q, model=model, index=index, items=items, top_k=args.top_k)
        for r in recs:
            rows.append(
                {
                    "Query": q,
                    "Assessment_url": r["url"],
                }
            )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)

    print(f"✅ Wrote submission CSV: {out_path}")
    print("Columns:", ["Query", "Assessment_url"])
    print(f"Rows: {len(rows)} (should be queries * top_k)")


if __name__ == "__main__":
    main()
