"""
evaluation/evaluate.py — read generation results, compute every metric per row.
Output: metrics.csv with one row per (pair x model x variant).
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import pandas as pd
sys.path.append(str(Path(__file__).resolve().parents[1]))
import config as C
from evaluation import metrics as M


def evaluate(results_path=None, out_csv=None):
    results_path = Path(results_path or (C.DATA_RESULTS / "generation_results.jsonl"))
    out_csv = Path(out_csv or (C.DATA_RESULTS / "metrics.csv"))
    rows = [json.loads(l) for l in open(results_path, encoding="utf-8") if l.strip()]
    rows = [r for r in rows if r.get("status") == "ok" and r.get("generated")]

    refs = [r["reference_doc"] for r in rows]
    hyps = [r["generated"] for r in rows]

    # batched metrics
    bert = M.bertscore_batch(refs, hyps)
    cos  = M.cosine_batch(refs, hyps)

    out = []
    for r, ref, hyp, b, c in zip(rows, refs, hyps, bert, cos):
        r1, rl = M.rouge(ref, hyp)
        out.append({
            "dataset": r["dataset"], "model": r["model"], "variant": r["variant"],
            "task_type": r["task_type"], "repo": r["repo"], "pair_id": r["pair_id"],
            "rouge1": r1, "rougeL": rl, "bleu": M.bleu(ref, hyp),
            "meteor": M.meteor(ref, hyp), "bertscore_f1": b, "cosine": c,
            "generated": hyp, "reference_doc": ref,
        })
    df = pd.DataFrame(out)
    df.to_csv(out_csv, index=False)
    print(f"Evaluated {len(df)} rows -> {out_csv}")
    return df
