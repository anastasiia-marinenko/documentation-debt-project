"""
scripts/llm_generation/apply_inline_doc_changes.py

EXP-001 v2 — Applies LLM-generated inline documentation to source files.

Takes the generated INLINE DOCUMENTATION sections from multi_llm_results.jsonl
and produces "before" / "after" source file versions for manual review and
comparison with the ground-truth patch.

Replaces the old apply_readme_changes.py which tried to apply updates to
README.md files — incorrect for the Lin et al. dataset, which contains
code file changes, not README changes.

Output per PR × model:
  data/results/applied_changes/
    pr{N}_{repo}_{lang}_before{ext}    — original source file
    pr{N}_{repo}_{model}_after{ext}    — source file with doc comment inserted
    pr{N}_{repo}_ground_truth.diff     — actual diff from the PR
"""

import json
import re
from pathlib import Path

import pandas as pd

# ============================================================
# CONFIG
# ============================================================

ROOT_DIR      = Path(__file__).resolve().parents[2]
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
# PARSE GENERATED OUTPUT
# ============================================================

def parse_generated(text: str) -> dict:
    """
    Extract sections from the v3 prompt output:
      INLINE DOCUMENTATION: ...
      PLACEMENT: ...
      REASON: ...
    Returns dict with keys: doc, placement, reason
    """
    if not text or text.startswith("ERROR") or text.startswith("SKIPPED"):
        return None

    def extract_section(label: str, next_labels: list) -> str:
        pattern = rf"{label}[:\s]+(.*?)(?={'|'.join(next_labels)}|$)"
        m = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        return m.group(1).strip() if m else ""

    doc       = extract_section("INLINE DOCUMENTATION", ["PLACEMENT", "REASON"])
    placement = extract_section("PLACEMENT", ["REASON"])
    reason    = extract_section("REASON", [])

    if not doc:
        return None

    return {"doc": doc, "placement": placement, "reason": reason}


def get_original_content(row) -> tuple[str, str]:
    """
    Returns (content, file_extension) of the source file before the PR.
    Tries readme_patches_json → old_content first, then reconstructs from diff.
    """
    lang = str(row.get("lang", "")).lower()
    ext  = lang if lang.startswith(".") else f".{lang}"

    # Strategy 1: old_content field
    try:
        patches = json.loads(row.get("readme_patches_json", "[]"))
        for p in patches:
            content = str(p.get("old_content", "")).strip()
            if content and content != "nan":
                return content, ext
    except Exception:
        pass

    # Strategy 2: reconstruct from diff (keep context + removed lines)
    try:
        patches = json.loads(row.get("readme_patches_json", "[]"))
        lines = []
        for p in patches:
            for line in str(p.get("patch", "")).splitlines():
                if line.startswith("@@"):
                    lines.append(f"// ... (diff hunk) ...")
                elif line.startswith("-") and not line.startswith("---"):
                    lines.append(line[1:])
                elif not line.startswith("+"):
                    lines.append(line)
        content = "\n".join(lines).strip()
        if content:
            return content, ext
    except Exception:
        pass

    return None, ext

# ============================================================
# APPLY CHANGES
# ============================================================

print("\n" + "=" * 60)
print("Generating before/after source file views...")
print("=" * 60)

summary = []

for pr_idx in df_results["pr_id"].unique():
    pr_results = df_results[df_results["pr_id"] == pr_idx]

    if pr_idx >= len(df_golden):
        continue
    row = df_golden.iloc[pr_idx]

    repo      = row.get("repo", f"pr_{pr_idx}")
    lang      = str(row.get("lang", ".txt")).lower()
    ext       = lang if lang.startswith(".") else f".{lang}"
    safe_repo = repo.replace("/", "_")

    original_content, ext = get_original_content(row)

    print(f"\nPR {pr_idx}: {repo}  ({lang})")

    # --- Ground truth diff ---
    gt_patch = ""
    try:
        patches = json.loads(row.get("readme_patches_json", "[]"))
        gt_patch = patches[0].get("patch", "") if patches else ""
    except Exception:
        pass

    gt_path = OUTPUT_DIR / f"pr{pr_idx:02d}_{safe_repo}_ground_truth.diff"
    gt_path.write_text(
        f"# Ground Truth Patch — {repo}\n# Language: {lang}\n\n{gt_patch}",
        encoding="utf-8",
    )

    # --- Before file ---
    if original_content:
        before_path = OUTPUT_DIR / f"pr{pr_idx:02d}_{safe_repo}_before{ext}"
        before_path.write_text(
            f"// Original file — {repo}\n\n{original_content}",
            encoding="utf-8",
        )
    else:
        print(f"  ⚠ No original content available for PR {pr_idx} — skipping before file")

    # --- After files (one per model) ---
    for _, res_row in pr_results.iterrows():
        model  = res_row["model"]
        status = res_row.get("status", "error")

        if status != "ok":
            print(f"  {model:<14} SKIPPED (status={status})")
            continue

        parsed = parse_generated(str(res_row["generated"]))
        if not parsed:
            print(f"  {model:<14} SKIPPED (could not parse output)")
            continue

        doc_comment = parsed["doc"]
        placement   = parsed["placement"] or "(see generated comment)"
        reason      = parsed["reason"] or ""

        # Build the "after" file:
        # If we have the original, append the doc comment with a clear separator.
        # (Surgical insertion requires AST parsing; appending is honest and reviewable.)
        if original_content:
            after_content = (
                original_content
                + f"\n\n"
                + f"// ─────────────────────────────────────────────────────\n"
                + f"// LLM-generated inline documentation ({model})\n"
                + f"// Suggested placement: {placement}\n"
                + f"// Reason: {reason}\n"
                + f"// ─────────────────────────────────────────────────────\n"
                + doc_comment
            )
        else:
            after_content = (
                f"// Original source not available — generated doc comment only\n"
                + f"// Suggested placement: {placement}\n"
                + f"// Reason: {reason}\n\n"
                + doc_comment
            )

        after_path = OUTPUT_DIR / f"pr{pr_idx:02d}_{safe_repo}_{model}_after{ext}"
        after_path.write_text(after_content, encoding="utf-8")
        print(f"  {model:<14} ✓  → {after_path.name}")

        summary.append({
            "pr_id":          pr_idx,
            "repo":           repo,
            "lang":           lang,
            "model":          model,
            "doc_comment":    doc_comment[:300],
            "placement":      placement,
            "reason":         reason,
            "has_original":   bool(original_content),
            "after_file":     str(after_path),
            "gt_diff":        str(gt_path),
        })

# ============================================================
# SUMMARY CSV
# ============================================================

summary_df   = pd.DataFrame(summary)
summary_path = OUTPUT_DIR / "applied_changes_index.csv"
summary_df.to_csv(summary_path, index=False)

print("\n" + "=" * 60)
print("DONE — EXP-001 v2 apply step")
print("=" * 60)
print(f"Files written : {len(list(OUTPUT_DIR.iterdir()))}")
print(f"Summary CSV   : {summary_path}")
print(f"Output folder : {OUTPUT_DIR}")
print()
print("Review workflow:")
print("  1. Open pr*_before.* and pr*_*_after.* side by side")
print("  2. Compare the LLM-suggested doc comment with ground_truth.diff")
print("  3. Score in comparison_table.csv: format_correct / placement / correct / useful")