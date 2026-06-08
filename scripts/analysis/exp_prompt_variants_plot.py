"""
scripts/analysis/exp_prompt_variants_plot.py

EXP-003 — Prompt Engineering Comparison — Visualisation
=========================================================
Loads prompt_variants_metrics.csv and generates:
  Plot 1 — Grouped bar: ROUGE-1, ROUGE-L, BERTScore per variant
  Plot 2 — Delta vs baseline: how much each pattern improves over v2
  Plot 3 — Per-repo BERTScore heatmap: which repos benefit most from each prompt

Run after exp_prompt_variants.py:
    python scripts/analysis/exp_prompt_variants_plot.py

Input : data/results/prompt_variants_metrics.csv
Output: data/results/plots/exp003_prompt_variants_bar.png
        data/results/plots/exp003_prompt_variants_delta.png
        data/results/plots/exp003_prompt_variants_heatmap.png
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd

# ============================================================
# CONFIG
# ============================================================

ROOT       = Path(__file__).resolve().parents[2]
INPUT_CSV  = ROOT / "data/results/prompt_variants_metrics.csv"
OUTPUT_DIR = ROOT / "data/results/plots"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# variant order and display labels
VARIANT_ORDER = [
    "v1_minimal",
    "v2_structured",
    "v3_template",
    "v4_persona",
    "v5_cognitive",
]

VARIANT_LABELS = {
    "v1_minimal":   "v1\nMinimal",
    "v2_structured":"v2\nStructured\n(baseline)",
    "v3_template":  "v3\nTemplate\nPattern",
    "v4_persona":   "v4\nPersona +\nTemplate",
    "v5_cognitive": "v5\nCognitive\nVerifier",
}

# colour per variant (sequential blue palette)
VARIANT_COLORS = {
    "v1_minimal":    "#AECDE8",
    "v2_structured": "#4A90E2",
    "v3_template":   "#1F5FA6",
    "v4_persona":    "#27AE60",
    "v5_cognitive":  "#E67E22",
}

BASELINE = "v2_structured"

# ============================================================
# LOAD
# ============================================================

print("=" * 60)
print("EXP-003 Prompt Variants — Plotting")
print("=" * 60)

if not INPUT_CSV.exists():
    print(f"ERROR: {INPUT_CSV} not found.")
    print("Run scripts/llm_generation/exp_prompt_variants.py first.")
    exit(1)

df = pd.read_csv(INPUT_CSV)
print(f"Loaded {len(df)} rows")
print(f"Variants found: {df['variant'].unique().tolist()}")

ok = df[df["status"] == "ok"].copy()
for col in ["rouge1", "rougeL", "bertscore"]:
    ok[col] = pd.to_numeric(ok[col], errors="coerce")

# aggregate per variant
agg = ok.groupby("variant")[["rouge1", "rougeL", "bertscore"]].mean().round(4)

# reorder to match VARIANT_ORDER (keep only variants that exist)
order = [v for v in VARIANT_ORDER if v in agg.index]
agg   = agg.loc[order]

print("\nMean metrics per variant:")
print(agg.to_string())

# ============================================================
# PLOT 1 — Grouped bar: all three metrics side by side
# ============================================================

metrics     = ["rouge1", "rougeL", "bertscore"]
metric_lbls = ["ROUGE-1", "ROUGE-L", "BERTScore F1"]
x           = np.arange(len(metrics))
width       = 0.15
n           = len(order)

fig, ax = plt.subplots(figsize=(12, 6))
fig.patch.set_facecolor("#F8F9FA")
ax.set_facecolor("#F8F9FA")

for i, variant in enumerate(order):
    vals   = [agg.loc[variant, m] for m in metrics]
    offset = (i - n / 2 + 0.5) * width
    bars   = ax.bar(
        x + offset, vals, width,
        label=VARIANT_LABELS.get(variant, variant).replace("\n", " "),
        color=VARIANT_COLORS.get(variant, "#888"),
        alpha=0.88, edgecolor="white", linewidth=0.8,
    )
    for bar, v in zip(bars, vals):
        if not np.isnan(v):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{v:.3f}",
                ha="center", va="bottom", fontsize=7.5, color="#333",
            )

ax.set_xticks(x)
ax.set_xticklabels(metric_lbls, fontsize=12)
ax.set_ylabel("Score", fontsize=12)
ax.set_ylim(0, max(agg.values.max() * 1.15, 0.5))
ax.set_title(
    "EXP-003: Prompt Variants Comparison (v1–v5)\n"
    "Model fixed: llama-3.1-8b-instant | Dataset: 25 Java pairs",
    fontsize=13, fontweight="bold", pad=14,
)
ax.legend(
    fontsize=9, loc="upper right",
    framealpha=0.9, edgecolor="#ccc",
)
ax.grid(axis="y", alpha=0.35, linestyle="--")
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()

out1 = OUTPUT_DIR / "exp003_prompt_variants_bar.png"
plt.savefig(out1, dpi=150, bbox_inches="tight")
plt.close()
print(f"\nPlot saved: {out1}")

# ============================================================
# PLOT 2 — Delta vs baseline (v2_structured)
# Shows the actual gain or loss from each pattern
# ============================================================

if BASELINE in agg.index:
    base = agg.loc[BASELINE]
    delta_df = (agg - base).drop(index=BASELINE)

    variants_d = delta_df.index.tolist()
    x2         = np.arange(len(variants_d))
    width2     = 0.22

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("#F8F9FA")
    ax.set_facecolor("#F8F9FA")

    for j, (metric, label) in enumerate(zip(metrics, metric_lbls)):
        vals   = delta_df[metric].tolist()
        offset = (j - len(metrics) / 2 + 0.5) * width2
        colors = ["#27AE60" if v >= 0 else "#E74C3C" for v in vals]
        bars   = ax.bar(
            x2 + offset, vals, width2,
            label=label,
            color=colors,
            alpha=0.80, edgecolor="white", linewidth=0.8,
        )
        for bar, v in zip(bars, vals):
            if not np.isnan(v):
                va  = "bottom" if v >= 0 else "top"
                yp  = bar.get_height() + 0.002 if v >= 0 else bar.get_height() - 0.002
                ax.text(
                    bar.get_x() + bar.get_width() / 2, yp,
                    f"{v:+.3f}",
                    ha="center", va=va, fontsize=8, color="#333",
                )

    ax.axhline(0, color="#555", linewidth=1.0, linestyle="-")
    ax.set_xticks(x2)
    ax.set_xticklabels(
        [VARIANT_LABELS.get(v, v).replace("\n", " ") for v in variants_d],
        fontsize=10,
    )
    ax.set_ylabel("Delta vs v2_structured baseline", fontsize=11)
    ax.set_title(
        "EXP-003: Prompt Pattern Effect — Delta vs Baseline (v2)\n"
        "Green = improvement | Red = regression",
        fontsize=12, fontweight="bold", pad=12,
    )
    ax.legend(fontsize=10, loc="upper right", framealpha=0.9)
    ax.grid(axis="y", alpha=0.35, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()

    out2 = OUTPUT_DIR / "exp003_prompt_variants_delta.png"
    plt.savefig(out2, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved: {out2}")

# ============================================================
# PLOT 3 — Per-repo BERTScore heatmap
# Shows which repositories benefit most from specific patterns
# ============================================================

if "repo" in ok.columns:
    pivot = ok.pivot_table(
        index="repo", columns="variant", values="bertscore", aggfunc="mean"
    )
    # keep only variants we ordered, drop missing
    pivot = pivot[[v for v in order if v in pivot.columns]]
    # shorten repo names (take last two path components)
    pivot.index = ["/".join(r.split("/")[-2:]) if "/" in r else r
                   for r in pivot.index]
    pivot = pivot.sort_values(by=order[-1], ascending=False)

    if len(pivot) >= 2:
        fig, ax = plt.subplots(
            figsize=(max(8, len(order) * 1.8), max(5, len(pivot) * 0.45))
        )
        fig.patch.set_facecolor("#F8F9FA")

        im = ax.imshow(
            pivot.values, aspect="auto",
            cmap="RdYlGn", vmin=0.75, vmax=0.90,
        )

        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(
            [VARIANT_LABELS.get(v, v).replace("\n", " ") for v in pivot.columns],
            fontsize=9,
        )
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index, fontsize=8)

        # annotate cells
        for r in range(len(pivot.index)):
            for c in range(len(pivot.columns)):
                val = pivot.values[r, c]
                if not np.isnan(val):
                    ax.text(
                        c, r, f"{val:.3f}",
                        ha="center", va="center",
                        fontsize=8,
                        color="black" if 0.78 < val < 0.88 else "white",
                    )

        plt.colorbar(im, ax=ax, label="BERTScore F1", shrink=0.8)
        ax.set_title(
            "EXP-003: BERTScore per Repository × Prompt Variant\n"
            "Darker green = higher similarity to ground truth",
            fontsize=12, fontweight="bold", pad=12,
        )
        plt.tight_layout()

        out3 = OUTPUT_DIR / "exp003_prompt_variants_heatmap.png"
        plt.savefig(out3, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Plot saved: {out3}")

# ============================================================
# FINDINGS SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("FINDINGS SUMMARY — EXP-003 Prompt Engineering")
print("=" * 60)

best_rouge  = agg["rouge1"].idxmax()
best_bert   = agg["bertscore"].idxmax()

print(f"\nBest ROUGE-1   : {best_rouge}  ({agg.loc[best_rouge,'rouge1']:.4f})")
print(f"Best BERTScore : {best_bert}  ({agg.loc[best_bert,'bertscore']:.4f})")

if BASELINE in agg.index:
    print(f"\nImprovements over {BASELINE}:")
    for v in order:
        if v == BASELINE:
            continue
        d = agg.loc[v, "rouge1"] - agg.loc[BASELINE, "rouge1"]
        db = agg.loc[v, "bertscore"] - agg.loc[BASELINE, "bertscore"]
        sign = "↑" if d > 0 else "↓"
        print(f"  {v:<22} ROUGE-1 {d:+.4f} {sign}   BERTScore {db:+.4f}")

print("""
Pattern linkage (White et al., 2023):
  v3 Template Pattern   — explicit output rules reduce verbosity
  v4 Persona Pattern    — role priming improves precision
  v5 Cognitive Verifier — step-by-step reasoning reduces hallucination
  v4_meta Meta-Prompting (Suzgun & Kalai, 2024) — see meta_prompt_comparison.csv
""")
print(f"All plots saved to: {OUTPUT_DIR}")
