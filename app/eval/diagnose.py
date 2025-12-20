from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd

from app.index.search_index import (
    canonicalize_url,
    load_artifacts,
    recommend,
    slug_from_url,
)

DEBUG_DIR = Path("data/debug")
DEBUG_DIR.mkdir(parents=True, exist_ok=True)


def load_ground_truth_train(xlsx_path: str) -> Dict[str, List[str]]:
    df = pd.read_excel(xlsx_path, sheet_name="Train-Set")
    gt: Dict[str, List[str]] = {}
    for q in df["Query"].unique():
        q = str(q).strip()
        urls = df[df["Query"] == q]["Assessment_url"].tolist()
        gt[q] = [canonicalize_url(str(u)) for u in urls]
    return gt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--xlsx", required=True)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--top-n", type=int, default=150)
    args = ap.parse_args()

    gt = load_ground_truth_train(args.xlsx)
    print(f"Loaded {len(gt)} Train-Set queries")

    artifacts = load_artifacts()

    rows = []
    per_query = []

    for q, truth_urls in gt.items():
        # get more candidates by asking for larger top_k (we'll slice later)
        recs = recommend(
            query=q,
            model=artifacts.model,
            index=artifacts.faiss_index,
            bm25=artifacts.bm25,
            items=artifacts.items,
            top_k=max(args.top_n, args.k),
        )

        pred_urls = [r["url"] for r in recs]
        pred_slugs = [slug_from_url(u) for u in pred_urls]
        truth_slugs = [slug_from_url(u) for u in truth_urls]

        pred_topk = set(s for s in pred_slugs[: args.k] if s)
        truth_set = set(s for s in truth_slugs if s)
        hits = sorted(list(pred_topk & truth_set))

        recall = (len(hits) / len(truth_set)) if truth_set else 0.0

        rank_map = {}
        for t in truth_set:
            try:
                rank_map[t] = pred_slugs.index(t) + 1
            except ValueError:
                rank_map[t] = None

        out = {
            "query": q,
            "recall": recall,
            "truth_slugs": sorted(list(truth_set)),
            "pred_slugs_topk": [s for s in pred_slugs[: args.k] if s],
            "hits": hits,
            "truth_ranks_within_topN": rank_map,
        }

        fname = DEBUG_DIR / f"query_{abs(hash(q))}.json"
        with open(fname, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)

        rows.append(
            {
                "query": q,
                "recall": recall,
                "truth_slugs": " | ".join(sorted(list(truth_set))),
                "pred_slugs_topk": " | ".join([s for s in pred_slugs[: args.k] if s]),
                "hits": " | ".join(hits),
                "truth_ranks_within_topN": json.dumps(rank_map),
                "per_query_json": str(fname),
            }
        )
        per_query.append((recall, q, str(fname)))

    mean_recall = sum(r["recall"] for r in rows) / max(len(rows), 1)
    print("\n====================")
    print(f"Mean Recall@{args.k}: {mean_recall:.4f}")
    print("====================\n")

    per_query.sort(key=lambda x: x[0])
    print("Worst 5 queries:")
    for r, q, f in per_query[:5]:
        print(f"- Recall={r:.3f} | {q[:90]}{'...' if len(q) > 90 else ''}")
        print(f"  -> {f}")

    out_csv = DEBUG_DIR / "diagnosis_summary.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"\nWrote: {out_csv}")
    print(f"Wrote per-query JSONs into: {DEBUG_DIR}")


if __name__ == "__main__":
    main()
