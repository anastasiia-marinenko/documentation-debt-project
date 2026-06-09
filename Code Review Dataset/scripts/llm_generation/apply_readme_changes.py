"""
scripts/llm_generation/apply_readme_changes.py

Takes LLM-generated README updates from multi_llm_results.jsonl
and applies them to the original README file (oldf column), producing
a modified README that can be compared to the ground truth (oldf + patch).

Output:
  data/results/applied_changes/
    pr_{id}_{model}_before.md    — original README
    pr_{id}_{model}_after.md     — README with LLM update appended/inserted
    pr_{id}_ground_truth.md      — actual README after PR (oldf + patch applied)
"""

import json
import re
import subprocess
import tempfile
from pathlib import Path

import pandas as pd

# ============================================================
# CONFIG
# ============================================================

ROOT_DIR    = Path(__file__).resolve().parents[2]
RESULTS_JSONL = ROOT_DIR / "data/results/multi_llm_results.jsonl"
GOLDEN_SET    = ROOT_DIR / "data/processed/golden_set.parquet"
OUTPUT_DIR    = ROOT_DIR / "data/results/applied_changes"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# LOAD
# ============================================================

print("=" * 60)
print("Loading results and golden set...")
print("=" * 60)

if not RESULTS_JSONL.exists():
    print(f"ERROR: {RESULTS_JSONL} not found.")
    print("Run generate_docs_multi_llm.py first.")
    exit(1)

results = []
with open(RESULTS_JSONL, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            results.append(json.loads(line))

df_results = pd.DataFrame(results)
df_golden  = pd.read_parquet(GOLDEN_SET)

print(f"Results rows : {len(df_results)}")
print(f"Golden set   : {len(df_golden)} PRs")

# ============================================================
# EXTRACT GENERATED TEXT
# ============================================================

def extract_update_text(generated: str) -> str:
    """Pull out just the DOCUMENTATION UPDATE section."""
    if generated.startswith("ERROR") or generated.startswith("SKIPPED"):
        return None
    # Look for the DOCUMENTATION UPDATE block
    m = re.search(
        r"DOCUMENTATION UPDATE[:\s]+(.*?)(?:REASON[:\s]+|$)",
        generated,
        re.DOTALL | re.IGNORECASE,
    )
    if m:
        return m.group(1).strip()
    # Fallback: return whole text
    return generated.strip()

# ============================================================
# APPLY CHANGES
# ============================================================

print("\n" + "=" * 60)
print("Applying LLM-generated changes to README files...")
print("=" * 60)

summary = []

for pr_idx in df_results["pr_id"].unique():
    pr_results = df_results[df_results["pr_id"] == pr_idx]

    # Get original README content from golden set
    if pr_idx >= len(df_golden):
        continue
    row = df_golden.iloc[pr_idx]

    # Original README (before PR)
    # Strategy 1: direct oldf column
    readme_before = str(row.get("oldf", "")).strip()
    if not readme_before or readme_before == "nan":
        # Strategy 2: old_content from readme_patches_json
        try:
            import json as _json
            patches = _json.loads(row.get("readme_patches_json", "[]"))
            for p in patches:
                old_content = str(p.get("old_content", "")).strip()
                if old_content:
                    readme_before = old_content
                    break
        except Exception:
            pass
    if not readme_before or readme_before == "nan":
        # Strategy 3: reconstruct from patch by stripping diff markers
        try:
            import json as _json
            patches = _json.loads(row.get("readme_patches_json", "[]"))
            reconstructed = []
            for p in patches:
                for line in str(p.get("patch", "")).splitlines():
                    if line.startswith("-") and not line.startswith("---"):
                        reconstructed.append(line[1:])
                    elif not line.startswith("+") and not line.startswith("@@"):
                        reconstructed.append(line)
            if reconstructed:
                readme_before = "\n".join(reconstructed).strip()
        except Exception:
            pass
    if not readme_before or readme_before == "nan":
        readme_before = "(original README not available in this dataset)"
        print(f"  ⚠ WARNING: no original README found for PR {pr_idx} — skipping apply step")

    repo = row.get("repo", f"pr_{pr_idx}")
    safe_repo = repo.replace("/", "_")

    # Save original README (before PR)
    before_path = OUTPUT_DIR / f"pr{pr_idx:02d}_{safe_repo}_before.md"
    before_path.write_text(
        f"# Original README — {repo}\n\n{readme_before}",
        encoding="utf-8",
    )

    if readme_before == "(original README not available in this dataset)":
        print(f"  ⚠ Skipping LLM apply for PR {pr_idx} — no source README to patch")
        continue

    # Ground truth patch (what actually changed)
    gt_patch = str(row.get("ground_truth_patch", ""))
    gt_path  = OUTPUT_DIR / f"pr{pr_idx:02d}_{safe_repo}_ground_truth.diff"
    gt_path.write_text(
        f"# Ground Truth Patch — {repo}\n\n{gt_patch}",
        encoding="utf-8",
    )

    print(f"\nPR {pr_idx}: {repo}")

    for _, res_row in pr_results.iterrows():
        model  = res_row["model"]
        status = res_row["status"]

        if status != "ok":
            print(f"  {model:<14} SKIPPED (status={status})")
            continue

        update_text = extract_update_text(str(res_row["generated"]))
        if not update_text:
            print(f"  {model:<14} SKIPPED (no update extracted)")
            continue

        # Apply: append generated update to end of README
        # (simple strategy — surgical insertion would require more complex parsing)
        readme_after = (
            readme_before
            + "\n\n---\n"
            + f"<!-- LLM-generated update ({model}) -->\n"
            + update_text
        )

        after_path = OUTPUT_DIR / f"pr{pr_idx:02d}_{safe_repo}_{model}_after.md"
        after_path.write_text(readme_after, encoding="utf-8")

        print(f"  {model:<14} ✓  → {after_path.name}")

        summary.append({
            "pr_id":       pr_idx,
            "repo":        repo,
            "model":       model,
            "update_text": update_text[:200],
            "before_file": str(before_path),
            "after_file":  str(after_path),
            "gt_file":     str(gt_path),
        })

# ============================================================
# SUMMARY CSV
# ============================================================

summary_df = pd.DataFrame(summary)
summary_path = OUTPUT_DIR / "applied_changes_index.csv"
summary_df.to_csv(summary_path, index=False)

print("\n" + "=" * 60)
print("DONE")
print("=" * 60)
print(f"Files written : {len(list(OUTPUT_DIR.iterdir()))}")
print(f"Summary CSV   : {summary_path}")
print(f"Output folder : {OUTPUT_DIR}")
print("\nNext: open the *_before.md and *_after.md files side by side")
print("and compare with the ground_truth.diff to evaluate quality.")