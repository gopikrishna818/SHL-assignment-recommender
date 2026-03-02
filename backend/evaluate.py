"""
evaluate.py — I wrote this to measure how well the recommender is doing.

I compute Recall@10 on the labeled train queries and generate predictions
for the test set in the submission CSV format.

Usage:
  python evaluate.py --mode train    # check recall on train set
  python evaluate.py --mode test     # generate test_predictions.csv
  python evaluate.py --mode both     # do both (what I usually run)
"""

import argparse
import json
import csv
import os
import time
import requests
import pandas as pd
from collections import defaultdict

# I added a small delay between API calls because Groq has rate limits
# on the free tier. 8 seconds works fine in practice.
API_CALL_DELAY = 8


def normalize_url(url: str) -> str:
    # I compare by slug (last path segment) because SHL URLs sometimes have
    # slightly different prefixes depending on where they come from
    if not url:
        return ""
    return str(url).strip().rstrip("/").split("/")[-1].lower()


def call_api(api_base: str, query: str, retries: int = 3) -> list[str]:
    # Hits the /recommend endpoint and returns the list of predicted URLs.
    # I retry up to 3 times in case of a timeout or transient error.
    for attempt in range(retries):
        try:
            resp = requests.post(
                f"{api_base}/recommend",
                json={"query": query},
                timeout=120,
            )
            if resp.status_code == 200:
                data = resp.json()
                urls = [r["url"] for r in data.get("recommended_assessments", [])]
                if not urls:
                    print("    ⚠ API returned 200 but no results")
                return urls
            else:
                print(f"    ✗ API error {resp.status_code}: {resp.text[:120]}")
        except requests.exceptions.Timeout:
            print(f"    ✗ Timeout (attempt {attempt + 1}/{retries})")
            if attempt < retries - 1:
                time.sleep(10)
        except Exception as e:
            print(f"    ✗ Error: {e}")
            if attempt < retries - 1:
                time.sleep(5)
    return []


def recall_at_k(predicted_urls: list[str], relevant_urls: list[str], k: int = 10) -> float:
    # Standard Recall@K — what fraction of the relevant items did I find?
    # Using slug comparison to handle URL prefix inconsistencies
    if not relevant_urls:
        return 0.0
    pred_slugs = {normalize_url(u) for u in predicted_urls[:k]}
    rel_slugs  = {normalize_url(u) for u in relevant_urls}
    return len(pred_slugs & rel_slugs) / len(rel_slugs)


def load_train(train_path: str) -> pd.DataFrame:
    # train.csv has Query and Assessment_url columns (multiple rows per query)
    return pd.read_csv(train_path)


def load_test(test_path: str) -> pd.DataFrame:
    # test.csv has just the Query column
    return pd.read_csv(test_path)


# ─────────────────────────────────────────────────────────────────────────────
# Train evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_train(
    api_base: str,
    train_df: pd.DataFrame,
    k: int = 10,
    resume: bool = False,
) -> float:
    print("=" * 60)
    print(f"Evaluating on Train Set  (Recall@{k})")
    print(f"API: {api_base}")
    print("=" * 60)

    # Group all relevant URLs by query
    query_relevant: dict[str, list[str]] = defaultdict(list)
    for _, row in train_df.iterrows():
        query_relevant[row["Query"]].append(row["Assessment_url"])
    queries = list(query_relevant.keys())
    print(f"Total queries: {len(queries)}")

    out_path = os.path.join("evaluation", f"train_results_recall_at_{k}.json")
    os.makedirs("evaluation", exist_ok=True)

    # If I'm resuming a crashed run, load whatever got saved before
    completed: dict[str, dict] = {}
    if resume and os.path.exists(out_path):
        with open(out_path) as f:
            existing = json.load(f)
        for d in existing.get("details", []):
            completed[d["query"]] = d
        print(f"Resuming: {len(completed)} / {len(queries)} already done")

    recall_scores: list[float] = []
    results_detail: list[dict] = []

    for i, query in enumerate(queries):
        relevant = query_relevant[query]
        print(f"\n[{i + 1}/{len(queries)}] {query[:80]}...")
        print(f"  Relevant: {len(relevant)} assessments")

        if query in completed:
            cached = completed[query]
            score  = cached.get(f"recall_at_{k}", 0.0)
            recall_scores.append(score)
            results_detail.append(cached)
            print(f"  (Cached) Recall@{k}: {score:.4f}")
            continue

        predicted = call_api(api_base, query)
        score     = recall_at_k(predicted, relevant, k)
        recall_scores.append(score)
        print(f"  Predicted: {len(predicted)} | Recall@{k}: {score:.4f}")

        # Show which relevant items were hit and which were missed
        pred_slugs = {normalize_url(u) for u in predicted[:k]}
        for rel_url in relevant:
            slug = normalize_url(rel_url)
            hit  = "✅" if slug in pred_slugs else "❌"
            print(f"    {hit} {slug}")

        detail = {
            "query":            query,
            "relevant_count":   len(relevant),
            "predicted_count":  len(predicted),
            f"recall_at_{k}":   score,
            "predicted_urls":   predicted,
            "relevant_urls":    relevant,
        }
        results_detail.append(detail)

        # Save after every query so a crash doesn't lose progress
        _save_train_results(out_path, recall_scores, results_detail, k, i + 1)

        if i < len(queries) - 1:
            print(f"  ⏳ Waiting {API_CALL_DELAY}s...")
            time.sleep(API_CALL_DELAY)

    mean_recall = sum(recall_scores) / len(recall_scores) if recall_scores else 0.0

    print("\n" + "=" * 60)
    print(f"Mean Recall@{k}: {mean_recall:.4f}")
    print(f"Per query:       {[f'{s:.3f}' for s in recall_scores]}")
    print("=" * 60)

    _save_train_results(out_path, recall_scores, results_detail, k, len(queries))
    print(f"Saved → {out_path}")
    return mean_recall


def _save_train_results(path, scores, details, k, queries_done):
    mean = sum(scores) / len(scores) if scores else 0.0
    with open(path, "w") as f:
        json.dump(
            {
                f"mean_recall_at_{k}": mean,
                "individual_scores":   scores,
                "queries_completed":   queries_done,
                "details":             details,
            },
            f,
            indent=2,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Test predictions
# ─────────────────────────────────────────────────────────────────────────────

def generate_test_predictions(
    api_base: str,
    test_df: pd.DataFrame,
    out_csv: str = "evaluation/test_predictions.csv",
) -> list[dict]:
    print("\n" + "=" * 60)
    print("Generating Test Predictions")
    print("=" * 60)

    query_col = "Query" if "Query" in test_df.columns else test_df.columns[0]
    queries   = test_df[query_col].dropna().unique().tolist()
    print(f"Test queries: {len(queries)}")

    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)

    # Resume from existing CSV if it's there
    rows: list[dict]       = []
    done_queries: set[str] = set()
    if os.path.exists(out_csv):
        with open(out_csv, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
                done_queries.add(row["Query"])
        print(f"Resuming: {len(done_queries)} queries already predicted")

    for i, query in enumerate(queries):
        if query in done_queries:
            print(f"[{i + 1}/{len(queries)}] (Cached) {query[:80]}...")
            continue

        print(f"\n[{i + 1}/{len(queries)}] {query[:100]}...")
        predicted = call_api(api_base, query)
        print(f"  → {len(predicted)} predictions")

        if not predicted:
            print("  ⚠ No predictions — check API is running")

        for url in predicted:
            rows.append({"Query": query, "Assessment_url": url})

        # Write CSV after each query so I don't lose data if something crashes
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["Query", "Assessment_url"])
            writer.writeheader()
            writer.writerows(rows)

        if i < len(queries) - 1:
            print(f"  ⏳ Waiting {API_CALL_DELAY}s...")
            time.sleep(API_CALL_DELAY)

    print(f"\n✅ Test predictions saved → {out_csv}  ({len(rows)} rows)")
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate SHL Recommender")
    parser.add_argument("--api",   default="http://localhost:8000", help="API base URL")
    parser.add_argument("--train", default="data/train.csv",        help="Path to train CSV")
    parser.add_argument("--test",  default="data/test.csv",         help="Path to test CSV")
    parser.add_argument("--mode",  choices=["train", "test", "both"], default="both")
    parser.add_argument("--k",     type=int, default=10,             help="K for Recall@K")
    parser.add_argument("--out",   default="evaluation/test_predictions.csv",
                                                                     help="Output CSV path")
    parser.add_argument("--resume", action="store_true", help="Skip already-done queries")
    args = parser.parse_args()

    # Make sure the API is actually running before doing anything
    try:
        resp = requests.get(f"{args.api}/health", timeout=10)
        assert resp.json()["status"] == "healthy"
        print(f"✅ API healthy at {args.api}")
    except Exception as e:
        print(f"❌ API not reachable at {args.api}: {e}")
        return

    if args.mode in ("train", "both"):
        if not os.path.exists(args.train):
            print(f"❌ Train file not found: {args.train}")
            return
        train_df = load_train(args.train)
        print(f"✅ Loaded train set: {len(train_df)} rows")
        evaluate_train(args.api, train_df, args.k, resume=args.resume)

    if args.mode in ("test", "both"):
        if not os.path.exists(args.test):
            print(f"❌ Test file not found: {args.test}")
            return
        test_df = load_test(args.test)
        print(f"✅ Loaded test set: {len(test_df)} rows")
        generate_test_predictions(args.api, test_df, args.out)


if __name__ == "__main__":
    main()
