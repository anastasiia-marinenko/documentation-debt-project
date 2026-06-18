"""
compute_ira.py — inter-rater agreement (Cohen's kappa) for the 2-rater Javadoc eval.

Place in unified_pipeline/ira/  (or run standalone).  Run:  python compute_ira.py

Computes, PER METRIC (correctness, relevance, informativeness, fluency), on the
rows BOTH raters scored (the only valid basis for kappa):
  - Cohen's kappa: unweighted, linear-weighted, quadratic-weighted
  - exact-agreement % and within-1 %  (context for the "kappa paradox")
  - each rater's mean and the mean gap (leniency direction)
Plus each rater's bias vs the FINAL consensus (uses FINAL_RATING).

Ordinal note: scores are 1-5 Likert -> report QUADRATIC weighted kappa as primary
(a 4-vs-5 disagreement must not count like 1-vs-5). Unweighted + %agree are shown
for transparency, important because the data is skewed high (kappa paradox).
"""
from __future__ import annotations
import sys
import pandas as pd
from sklearn.metrics import cohen_kappa_score

XLSX     = sys.argv[1] if len(sys.argv) > 1 else "../data/raw/evaluation_384_final_2_raters.xlsx"
RATER_A  = "rater_Anastasiia"
RATER_N  = "rater_Nasser"
FINAL    = "FINAL_RATING"
METRICS  = ["correctness", "relevance", "informativeness", "fluency"]
LABELS   = [1, 2, 3, 4, 5]
ONLY_COMPLEXITY = None          # e.g. "simple" to restrict; None = all rated rows


def _num(df, m):
    return pd.to_numeric(df[m], errors="coerce")


def _pair(a_df, b_df, m):
    """Aligned, complete pairs for metric m (inner-join on ID, drop missing)."""
    j = pd.concat([_num(a_df, m).rename("a"), _num(b_df, m).rename("b")], axis=1).dropna()
    return j["a"].astype(int), j["b"].astype(int)


def main():
    A = pd.read_excel(XLSX, sheet_name=RATER_A).set_index("ID")
    N = pd.read_excel(XLSX, sheet_name=RATER_N).set_index("ID")
    F = pd.read_excel(XLSX, sheet_name=FINAL).set_index("ID")

    if ONLY_COMPLEXITY:
        keep = A.index[A["complexity_category"] == ONLY_COMPLEXITY]
        A, N, F = A.loc[A.index.isin(keep)], N.loc[N.index.isin(keep)], F.loc[F.index.isin(keep)]

    rows = []
    for m in METRICS:
        a, n = _pair(A, N, m)
        rows.append({
            "metric": m, "N": len(a),
            "kappa_unweighted": round(cohen_kappa_score(a, n, labels=LABELS), 3),
            "kappa_linear":     round(cohen_kappa_score(a, n, weights="linear", labels=LABELS), 3),
            "kappa_quadratic":  round(cohen_kappa_score(a, n, weights="quadratic", labels=LABELS), 3),
            "exact_%":  round((a.values == n.values).mean() * 100, 1),
            "within1_%": round((abs(a.values - n.values) <= 1).mean() * 100, 1),
            "mean_A": round(a.mean(), 2), "mean_N": round(n.mean(), 2),
        })
    res = pd.DataFrame(rows)

    # pooled across all four metrics
    aa = pd.concat([_pair(A, N, m)[0] for m in METRICS], ignore_index=True)
    nn = pd.concat([_pair(A, N, m)[1] for m in METRICS], ignore_index=True)
    pooled = {
        "metric": "POOLED", "N": len(aa),
        "kappa_unweighted": round(cohen_kappa_score(aa, nn, labels=LABELS), 3),
        "kappa_linear":     round(cohen_kappa_score(aa, nn, weights="linear", labels=LABELS), 3),
        "kappa_quadratic":  round(cohen_kappa_score(aa, nn, weights="quadratic", labels=LABELS), 3),
        "exact_%":  round((aa.values == nn.values).mean() * 100, 1),
        "within1_%": round((abs(aa.values - nn.values) <= 1).mean() * 100, 1),
        "mean_A": round(aa.mean(), 2), "mean_N": round(nn.mean(), 2),
    }
    res = pd.concat([res, pd.DataFrame([pooled])], ignore_index=True)

    pd.set_option("display.width", 200)
    print("=== Inter-rater agreement (rater_A vs rater_N, doubly-rated rows) ===")
    print(res.to_string(index=False))

    # bias of each rater vs consensus (this is what FINAL_RATING is for)
    print("\n=== Rater bias vs FINAL consensus (mean rater - mean final) ===")
    for m in METRICS:
        for nm, df in [("Anastasiia", A), ("Nasser", N)]:
            r, f = _pair(df, F, m)
            print(f"  {m:16s} {nm:10s} N={len(r):3d}  bias={(r.values - f.values).mean():+.2f}")

    res.to_csv("ira_kappa_results.csv", index=False)
    print("\nWrote ira_kappa_results.csv")
    print("Primary IRA measure = kappa_quadratic (ordinal 1-5). Interpret (Landis & Koch): "
          "<0.20 slight, 0.21-0.40 fair, 0.41-0.60 moderate, 0.61-0.80 substantial, >0.80 almost perfect.")


if __name__ == "__main__":
    main()
