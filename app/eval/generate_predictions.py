from __future__ import annotations

import argparse
import csv
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


@dataclass
class RecItem:
    name: str
    url: str
    description: str = ""


def load_test_queries(xlsx_path: str) -> List[str]:
    df = pd.read_excel(xlsx_path, sheet_name="Test-Set")
    # Try common column names
    for col in ["query", "Query", "prompt", "Prompt", "text", "Text"]:
        if col in df.columns:
            series = df[col].astype(str)
            break
    else:
        # fallback: first column
        series = df.iloc[:, 0].astype(str)

    queries = []
    for q in series.tolist():
        q = (q or "").strip()
        if q:
            queries.append(q)

    # Keep order, remove duplicates
    seen = set()
    uniq = []
    for q in queries:
        if q not in seen:
            uniq.append(q)
            seen.add(q)
    return uniq


def recommend_via_api(api_base: str, query: str, top_k: int, timeout_s: float = 30.0) -> List[RecItem]:
    import httpx

    payload = {"query": query, "top_k": int(top_k)}
    url = api_base.rstrip("/") + "/recommend"

    with httpx.Client(timeout=timeout_s) as client:
        r = client.post(url, json=payload)
        r.raise_for_status()
        data = r.json()

    items: List[RecItem] = []
    for x in data.get("recommended_assessments", []):
        name = str(x.get("name", "")).strip()
        url = str(x.get("url", "")).strip()
        desc = str(x.get("description", "") or "").strip()
        if url:
            items.append(RecItem(name=name, url=url, description=desc))
    return items


def recommend_locally(query: str, top_k: int) -> List[RecItem]:
    """
    Local mode is optional. It imports sentence-transformers/torch which may be broken on Windows.
    Keep it isolated so API mode still works.
    """
    from app.index.search_index import load_artifacts, recommend

    artifacts = load_artifacts()
    recs = recommend(query=query, top_k=top_k, artifacts=artifacts)

    items: List[RecItem] = []
    for r in recs:
        items.append(
            RecItem(
                name=str(r.get("name", "")).strip(),
                url=str(r.get("url", "")).strip(),
                description=str(r.get("description", "") or "").strip(),
            )
        )
    return [x for x in items if x.url]


def write_predictions_csv(out_path: str, rows: List[Dict[str, Any]]) -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["query", "rank", "url", "name"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--xlsx", required=True, help="Path to Gen_AI Dataset.xlsx")
    ap.add_argument("--k", type=int, default=10, help="Top-K recommendations per query")
    ap.add_argument("--out", required=True, help="Output CSV path")
    ap.add_argument("--api", default="", help="API base URL (recommended): e.g. http://127.0.0.1:8000")
    ap.add_argument("--retries", type=int, default=3, help="Retries per query (API mode)")
    ap.add_argument("--sleep", type=float, default=0.5, help="Sleep between queries (seconds)")
    args = ap.parse_args()

    xlsx_path = str(args.xlsx)
    top_k = int(args.k)
    out_path = str(args.out)
    api_base = (args.api or "").strip()

    queries = load_test_queries(xlsx_path)
    print(f"Loading Test-Set...\nLoaded {len(queries)} queries from Test-Set")

    rows: List[Dict[str, Any]] = []

    for i, q in enumerate(queries, start=1):
        rec_items: List[RecItem] = []
        err: Optional[Exception] = None

        if api_base:
            for attempt in range(1, args.retries + 1):
                try:
                    rec_items = recommend_via_api(api_base, q, top_k)
                    err = None
                    break
                except Exception as e:
                    err = e
                    time.sleep(1.0 * attempt)
            if err:
                print(f"[WARN] Query {i}/{len(queries)} failed after retries: {err}")
        else:
            try:
                rec_items = recommend_locally(q, top_k)
            except Exception as e:
                print(
                    "[ERROR] Local mode failed (likely torch DLL issue). "
                    "Run with --api http://127.0.0.1:8000 instead.\n"
                    f"Error: {e}"
                )
                rec_items = []

        # Write exactly K rows per query if possible
        rec_items = rec_items[:top_k]
        for rank, item in enumerate(rec_items, start=1):
            rows.append(
                {
                    "query": q,
                    "rank": rank,
                    "url": item.url,
                    "name": item.name,
                }
            )

        print(f"[{i:02d}/{len(queries)}] wrote {len(rec_items)} urls")
        time.sleep(args.sleep)

    write_predictions_csv(out_path, rows)
    print(f"\n✅ Wrote predictions CSV: {out_path}\n   Total rows: {len(rows)} (queries={len(queries)}; k={top_k})")


if __name__ == "__main__":
    main()
