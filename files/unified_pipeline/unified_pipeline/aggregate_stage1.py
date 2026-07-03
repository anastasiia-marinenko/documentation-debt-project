"""
aggregate_stage1.py — build stage1_best_wordings.json from per-method jsonl files.

Use this when you ran Stage 1 split BY PATTERN (--method zeroshot, --method oneshot, ...).
Each per-method run wrote data/results/prompt_select_v7_<model>_<method>.jsonl but only
saw its own method, so it could NOT pick the global best wording per pattern. This script
reads ALL those jsonl files, (re)scores them with BERTScore-F1, and writes the combined
data/results/stage1_best_wordings.json (best wording per (model, pattern)).

Run:  python aggregate_stage1.py
      python aggregate_stage1.py --model qwen_coder_14b      # restrict to one model
"""
from __future__ import annotations
import sys, json, glob, argparse, statistics, collections, random
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))
import config as C
from evaluation import metrics as M

random.seed(42)
STAGE1_BEST = C.DATA_RESULTS / "stage1_best_wordings.json"
SUMMARY_CSV = C.DATA_RESULTS / "prompt_select_v7_summary.csv"


def _bootstrap_ci(values, iters=2000, alpha=0.05):
    if len(values) < 2:
        v = values[0] if values else 0.0
        return (round(v, 4), round(v, 4))
    means = []
    for _ in range(iters):
        sample = [random.choice(values) for _ in values]
        means.append(statistics.mean(sample))
    means.sort()
    return (round(means[int((alpha / 2) * iters)], 4),
            round(means[int((1 - alpha / 2) * iters)], 4))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="restrict to one model key")
    args = ap.parse_args()

    pattern = f"prompt_select_v7_{args.model}_*.jsonl" if args.model else "prompt_select_v7_*.jsonl"
    files = sorted(glob.glob(str(C.DATA_RESULTS / pattern)))
    # keep only per-method files (they have a '_<method>' suffix); skip combined ones
    files = [f for f in files if "+meta" not in f]
    print(f"Reading {len(files)} jsonl files:")
    for f in files:
        print("  ", Path(f).name)

    rows = []
    for fp in files:
        for line in open(fp, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            # only Stage-1 rows = single patterns (no '+meta' method)
            if str(r.get("method", "")).endswith("+meta"):
                continue
            if r.get("status") == "ok" and r.get("generated"):
                rows.append(r)
    print(f"\nLoaded {len(rows)} successful generations.")
    if not rows:
        raise SystemExit("No successful rows found — check the jsonl files exist and have status=ok.")

    # score all at once (one BERTScore model load)
    f1 = M.bertscore_batch([r["reference_doc"] for r in rows],
                           [r["generated"] for r in rows])
    for r, s in zip(rows, f1):
        r["bertscore_f1"] = float(s)

    # aggregate per (model, method, prompt_id)
    agg = collections.defaultdict(list)
    for r in rows:
        agg[(r["model"], r["method"], r["prompt_id"])].append(r["bertscore_f1"])

    summ = [{"model": k[0], "method": k[1], "prompt_id": k[2], "n": len(v),
             "f1_mean": round(statistics.mean(v), 4),
             "f1_sd": round(statistics.pstdev(v), 4) if len(v) > 1 else 0.0,
             "f1_ci": _bootstrap_ci(v)}
            for k, v in agg.items()]
    summ.sort(key=lambda x: (x["model"], x["method"], -x["f1_mean"]))

    # write summary csv
    import csv
    with open(SUMMARY_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["model", "method", "prompt_id", "n",
                                           "f1_mean", "f1_sd", "f1_ci"])
        w.writeheader(); w.writerows(summ)
    print(f"Wrote {SUMMARY_CSV}")

    # best wording per (model, pattern)
    by_key = collections.defaultdict(list)
    for s in summ:
        by_key[(s["model"], s["method"])].append(s)
    best = {}
    for (mdl, pat), cand in by_key.items():
        w = max(cand, key=lambda x: x["f1_mean"])
        best.setdefault(mdl, {})[pat] = {"prompt_id": w["prompt_id"],
                                          "f1_mean": w["f1_mean"],
                                          "f1_ci": w.get("f1_ci"), "n": w["n"]}
    STAGE1_BEST.parent.mkdir(parents=True, exist_ok=True)
    with open(STAGE1_BEST, "w", encoding="utf-8") as f:
        json.dump(best, f, ensure_ascii=False, indent=2)
    print(f"Wrote {STAGE1_BEST}\n")
    for mdl, pats in best.items():
        print(mdl)
        for pat, info in pats.items():
            print(f"  {pat:10s} -> wording #{info['prompt_id']}  F1={info['f1_mean']}  (n={info['n']})")


if __name__ == "__main__":
    main()
