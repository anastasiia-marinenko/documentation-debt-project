"""
ira/compute_ira.py — inter-rater agreement once both rater sheets are filled.
Quadratic-weighted Cohen's kappa per ordinal item + Krippendorff alpha for overall.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import cohen_kappa_score
sys.path.append(str(Path(__file__).resolve().parents[1]))
import config as C

ITEMS = ["correct", "useful", "hallucination", "format_correct", "placement", "readability"]


def _alpha_ordinal(x, y):
    pairs = [(int(p), int(q)) for p, q in zip(x, y) if pd.notna(p) and pd.notna(q)]
    if len(pairs) < 2: return float("nan")
    vals = [v for pr in pairs for v in pr]; rng = (max(vals) - min(vals)) or 1
    Do = np.mean([((p - q) / rng) ** 2 for p, q in pairs])
    De = np.mean([((p - q) / rng) ** 2 for i, p in enumerate(vals) for q in vals[i+1:]])
    return 1 - Do / De if De else float("nan")


def compute(xlsx=None):
    xlsx = Path(xlsx or (C.DATA_RESULTS / "ira_eval_template.xlsx"))
    a = pd.read_excel(xlsx, "rater_Anastasiia").set_index("ira_id")
    n = pd.read_excel(xlsx, "rater_Nasser").set_index("ira_id")
    idx = a.index.intersection(n.index); a, n = a.loc[idx], n.loc[idx]
    print(f"Items compared: {len(idx)}\n{'item':16s}{'weighted_kappa':>16s}{'%exact':>9s}")
    ks = []
    for it in ITEMS:
        x, y = a[it], n[it]; m = x.notna() & y.notna()
        x, y = x[m].astype(int), y[m].astype(int)
        if len(x) == 0: continue
        k = 1.0 if x.nunique() == 1 and y.nunique() == 1 and (x == y).all() else \
            cohen_kappa_score(x, y, weights="quadratic", labels=[0, 1, 2])
        ks.append(k); print(f"{it:16s}{k:>16.3f}{(x==y).mean()*100:>8.0f}%")
    print(f"\noverall_0_10 Krippendorff alpha: {_alpha_ordinal(a['overall_0_10'], n['overall_0_10']):.3f}")
    if ks: print(f"mean weighted kappa: {np.mean(ks):.3f}")
    print("Landis & Koch: .2-.4 fair, .4-.6 moderate, .6-.8 substantial, .8-1 almost perfect")


if __name__ == "__main__":
    compute(sys.argv[1] if len(sys.argv) > 1 else None)
