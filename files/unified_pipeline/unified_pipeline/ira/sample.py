"""
ira/sample.py — stratified random sample of N items across datasets for IRA.
Both raters score the SAME items. Stratified = proportional per dataset so the
agreement estimate isn't dominated by one dataset. Single fixed task only.
"""
from __future__ import annotations
import json, sys, random
from pathlib import Path
import pandas as pd
sys.path.append(str(Path(__file__).resolve().parents[1]))
import config as C


def stratified_sample(metrics_csv=None, n=None, seed=None, one_model=None,
                      per_dataset=None, task=None):
    """Pick `per_dataset` items from EACH dataset (default 5 -> 5x5=25 for 5 datasets).
    task=None keeps every task type, so SATD datasets (TESORO) are included too."""
    metrics_csv = Path(metrics_csv or (C.DATA_RESULTS / "metrics.csv"))
    n = n or C.IRA_TOTAL_SAMPLES
    seed = seed if seed is not None else C.IRA_SEED
    df = pd.read_csv(metrics_csv)
    if one_model:                       # rate ONE model's output to keep it comparable
        df = df[df.model == one_model]
    if task:                            # optional: restrict to one task type
        df = df[df.task_type == task]
    random.seed(seed)

    datasets = sorted(df.dataset.unique())
    per = per_dataset or max(1, n // max(1, len(datasets)))
    picks = []
    for d in datasets:
        sub = df[df.dataset == d]
        picks.append(sub.sample(min(per, len(sub)), random_state=seed))
    out = pd.concat(picks).reset_index(drop=True)
    out.insert(0, "ira_id", range(len(out)))
    out.to_csv(C.DATA_RESULTS / "ira_sample.csv", index=False)
    print(f"IRA sample: {len(out)} items ({per}/dataset x {len(datasets)} datasets)")
    return out
