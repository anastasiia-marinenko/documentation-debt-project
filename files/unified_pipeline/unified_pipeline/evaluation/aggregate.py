"""
evaluation/aggregate.py — build the cross-dataset / cross-model / cross-prompt
comparison tables (the research deliverable) + a plot.
"""
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd
sys.path.append(str(Path(__file__).resolve().parents[1]))
import config as C

_METRICS = ["rouge1", "rougeL", "bleu", "meteor", "bertscore_f1", "cosine"]


def aggregate(metrics_csv=None, out_dir=None, xlsx_name="cross_comparison.xlsx"):
    metrics_csv = Path(metrics_csv or (C.DATA_RESULTS / "metrics.csv"))
    out_dir = Path(out_dir or C.DATA_RESULTS)
    df = pd.read_csv(metrics_csv)
    have = [m for m in _METRICS if m in df.columns and df[m].notna().any()]

    by_dataset = df.groupby("dataset")[have].mean().round(4)
    by_model   = df.groupby("model")[have].mean().round(4)
    by_variant = df.groupby("variant")[have].mean().round(4)
    by_dm      = df.groupby(["dataset", "model"])[have].mean().round(4)

    with pd.ExcelWriter(out_dir / xlsx_name) as xl:
        by_dataset.to_excel(xl, sheet_name="by_dataset")
        by_model.to_excel(xl, sheet_name="by_model")
        by_variant.to_excel(xl, sheet_name="by_prompt_variant")
        by_dm.to_excel(xl, sheet_name="by_dataset_x_model")
    by_dataset.to_csv(out_dir / "by_dataset.csv")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        ax = by_dataset[[m for m in ("bertscore_f1", "rougeL", "cosine") if m in have]].plot.bar()
        ax.set_ylabel("score"); ax.set_title("Metrics by dataset")
        plt.tight_layout(); plt.savefig(out_dir / "by_dataset.png", dpi=150); plt.close()
    except Exception as e:
        print("plot skipped:", e)

    print(f"Wrote {xlsx_name} + by_dataset.csv/png")
    return by_dataset
