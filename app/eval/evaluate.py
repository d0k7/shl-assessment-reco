from __future__ import annotations

import argparse
from collections import defaultdict
from typing import Dict, List, Tuple

import pandas as pd

from app.index.search_index import canonicalize_url, load_artifacts, slug_from_url, recommend


def load_ground_truth_train(xlsx_path: str) -> Dict[str, List[str]]:
    df = pd.read_excel(xlsx_path, sheet_name="Train-Set")
    gt: Dict[str, List[str]] = defaultdict(list)
    for _, row in df.iterrows():
        q = str(row["Query"]).strip()
        u = canonicalize_url(str(row["Assessment_url"]).strip())
        gt[q].append(u)
    return gt


def recall_at_k(pred_urls: List[str], truth_urls: List[str], k: int) -> float:
    pred_slugs = {slug_from_url(u) for u in pred_urls[:k]}
    truth_slugs = {slug_from_url(u) for u in truth_urls}
    pred_slugs.discard(None)
    truth_slugs.discard(None)
    if not truth_slugs:
        return 0.0
    return len(pred_slugs & truth_slugs) / len(truth_slugs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--xlsx", required=True)
    ap.add_argument("--k", type=int, default=10)
    args = ap.parse_args()

    print("Loading ground-truth Train-Set...")
    gt = load_ground_truth_train(args.xlsx)
    print(f"Loaded {len(gt)} unique queries from Train-Set")

    artifacts = load_artifacts()

    recalls: List[Tuple[float, str]] = []
    for q, truth_urls in gt.items():
        recs = recommend(
            query=q,
            model=artifacts.model,
            index=artifacts.faiss_index,
            bm25=artifacts.bm25,
            items=artifacts.items,
            top_k=args.k,
        )
        pred_urls = [r["url"] for r in recs]
        r = recall_at_k(pred_urls, truth_urls, args.k)
        recalls.append((r, q))

    mean_recall = sum(r for r, _ in recalls) / max(len(recalls), 1)

    print("\n====================")
    print(f"Mean Recall@{args.k}: {mean_recall:.4f}")
    print("====================\n")

    recalls.sort(key=lambda x: x[0])
    print("Bottom 5 queries by Recall:")
    for r, q in recalls[:5]:
        print(f"- Recall={r:.3f} | {q[:110]}{'...' if len(q) > 110 else ''}")


if __name__ == "__main__":
    main()
