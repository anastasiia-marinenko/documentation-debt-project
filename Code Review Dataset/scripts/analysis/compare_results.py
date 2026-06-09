"""
scripts/analysis/compare_results.py

Loads multi_llm_results.jsonl and prints a clean side-by-side
comparison of:
  - ground truth README patch
  - each model's generated output

Also computes simple automatic metrics (ROUGE if available, else overlap).
"""

import json
from pathlib import Path

import pandas as pd

# ============================================================
# CONFIG
# ============================================================

ROOT_DIR      = Path(__file__).resolve().parents[2]
RESULTS_JSONL = ROOT_DIR / "data/results/multi_llm_results.jsonl"

# ============================================================
# LOAD
# ============================================================

print("=" * 60)
print("Loading results...")
print("=" * 60)

if not RESULTS_JSONL.exists():
    print(f"ERROR: {RESULTS_JSONL} not found. Run generation script first.")
    exit(1)

records = []
with open(RESULTS_JSONL, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            records.append(json.loads(line))

df = pd.DataFrame(records)
ok_df = df[df["status"] == "ok"]

print(f"Total rows   : {len(df)}")
print(f"OK rows      : {len(ok_df)}")
print(f"Models found : {df['model'].unique().tolist()}")

# ============================================================
# SIMPLE TOKEN OVERLAP (if rouge_score not installed)
# ============================================================

def token_overlap(ref: str, hyp: str) -> float:
    """Simple unigram F1 between reference and hypothesis."""
    ref_tokens = set(ref.lower().split())
    hyp_tokens = set(hyp.lower().split())
    if not ref_tokens or not hyp_tokens:
        return 0.0
    common = ref_tokens & hyp_tokens
    precision = len(common) / len(hyp_tokens)
    recall    = len(common) / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)

def compute_rouge(ref: str, hyp: str):
    """Try ROUGE, fall back to token overlap."""
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
        scores = scorer.score(ref, hyp)
        return {
            "rouge1_f": round(scores["rouge1"].fmeasure, 3),
            "rougeL_f": round(scores["rougeL"].fmeasure, 3),
        }
    except ImportError:
        overlap = token_overlap(ref, hyp)
        return {
            "token_overlap_f1": round(overlap, 3),
        }

# ============================================================
# PRINT SIDE-BY-SIDE COMPARISON
# ============================================================

print("\n" + "=" * 60)
print("SIDE-BY-SIDE COMPARISON")
print("=" * 60)

auto_metrics = []

for pr_id in sorted(df["pr_id"].unique()):
    pr_rows = df[df["pr_id"] == pr_id]
    repo    = pr_rows["repo"].iloc[0]
    gt      = str(pr_rows["ground_truth_patch"].iloc[0])[:500]

    print(f"\n{'▓' * 60}")
    print(f"PR {pr_id}: {repo}")
    print(f"{'▓' * 60}")

    print("\n[GROUND TRUTH PATCH — what actually changed]\n")
    print(gt[:400])
    if len(gt) > 400:
        print("... (truncated)")

    for _, row in pr_rows[pr_rows["status"] == "ok"].iterrows():
        model  = row["model"]
        gen    = str(row["generated"])

        print(f"\n[{model.upper()} — generated]\n")
        print(gen[:500])
        if len(gen) > 500:
            print("... (truncated)")

        # Auto metric
        metrics = compute_rouge(gt, gen)
        metrics.update({"pr_id": pr_id, "repo": repo, "model": model})
        auto_metrics.append(metrics)

        metric_str = "  ".join(f"{k}={v}" for k, v in metrics.items()
                               if k not in ("pr_id", "repo", "model"))
        print(f"\n  Auto metric: {metric_str}")

# ============================================================
# AUTO METRICS SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("AUTOMATIC METRICS SUMMARY (per model)")
print("=" * 60)

metrics_df = pd.DataFrame(auto_metrics)
metric_cols = [c for c in metrics_df.columns if c not in ("pr_id", "repo", "model")]

if metric_cols:
    agg = metrics_df.groupby("model")[metric_cols].mean().round(3)
    print(agg.to_string())

    # Save
    out = ROOT_DIR / "data/results/auto_metrics.csv"
    metrics_df.to_csv(out, index=False)
    print(f"\nSaved: {out}")

print("\nNote: These are approximate automatic metrics.")
print("Manual evaluation in comparison_table.csv is the primary evaluation.")
print("\nDone.")