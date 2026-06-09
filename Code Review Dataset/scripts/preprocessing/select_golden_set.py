"""
scripts/preprocessing/select_golden_set.py

Selects 5-10 high-quality README PRs for the golden evaluation set.

HOW IT WORKS:
  1. Loads readme_prs.parquet (already filtered: README + code PRs)
  2. Streams raw JSONL once → collects README patches + old content for those PRs
  3. Scores each PR by quality criteria
  4. Prints top candidates with preview
  5. Saves golden_set.json + golden_set.parquet

NO re-running of build_pr_dataset.py needed.

WHAT MAKES A GOOD PR:
  - README patch has actual content changes (not just whitespace/blank lines)
  - Old README content exists (needed as ground truth baseline)
  - Code patches are also non-trivial
  - Commit message is not empty
"""

import json
from collections import defaultdict
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# ============================================================
# CONFIG
# ============================================================

ROOT_DIR      = Path(__file__).resolve().parents[2]
PARQUET_FILE  = ROOT_DIR / "data/processed/readme_prs.parquet"
RAW_JSONL     = ROOT_DIR / "data/raw/train_complete.jsonl"
OUTPUT_JSON   = ROOT_DIR / "data/processed/golden_set.json"
OUTPUT_PARQUET= ROOT_DIR / "data/processed/golden_set.parquet"

N_GOLDEN      = 10          # how many PRs to keep in final golden set
MIN_README_CHANGED_LINES = 3  # minimum meaningful lines changed in README

# ============================================================
# STEP 1 — load existing readme_prs.parquet
# ============================================================

print("=" * 60)
print("STEP 1: Loading readme_prs.parquet")
print("=" * 60)

df = pd.read_parquet(PARQUET_FILE)
print(f"Total README+code PRs available: {len(df):,}")

valid_ghids = set(df["ghid"].astype(str).tolist())
print(f"Unique PR IDs to look up: {len(valid_ghids):,}")

# ============================================================
# STEP 2 — stream raw JSONL, collect README file entries
# ============================================================

print("\n" + "=" * 60)
print("STEP 2: Streaming raw JSONL for README patches")
print("(Only collecting README file entries for our valid PRs)")
print("=" * 60)

# ghid -> list of readme file entries
readme_entries = defaultdict(list)

with open(RAW_JSONL, "r", encoding="utf-8") as f:
    for line in tqdm(f, desc="Scanning JSONL"):
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue

        ghid = str(row.get("ghid", ""))
        file_path = str(row.get("file", ""))

        # Only collect README file rows for our valid PRs
        if ghid not in valid_ghids:
            continue
        if not file_path.lower().endswith((
                ".md",
                ".rst",
                ".txt"
        )):
            continue

        readme_entries[ghid].append({
            "file":        file_path,
            "patch":       str(row.get("patch", "") or ""),
            "old_content": str(row.get("oldf",  "") or ""),
        })

print(f"\nPRs with README patch data found: {len(readme_entries):,}")

# ============================================================
# HELPER — count meaningful changed lines in a patch
# ============================================================

def count_meaningful_lines(patch_text: str) -> int:
    """
    Count lines that are actually added (+) or removed (-),
    ignoring diff headers (+++ / ---) and blank/whitespace-only lines.
    """
    if not patch_text:
        return 0
    changed = 0
    for line in patch_text.split("\n"):
        if line.startswith("+++") or line.startswith("---"):
            continue
        if line.startswith("+") or line.startswith("-"):
            content = line[1:].strip()
            if len(content) > 1:   # ignore blank lines and single-char
                changed += 1
    return changed

# ============================================================
# STEP 3 — score each PR
# ============================================================

print("\n" + "=" * 60)
print("STEP 3: Scoring PRs by quality")
print("=" * 60)

scored = []

for _, row in df.iterrows():
    ghid = str(row["ghid"])

    entries = readme_entries.get(ghid, [])
    if not entries:
        continue  # no README patch data found

    # --- README quality ---
    readme_changed = sum(
        count_meaningful_lines(e["patch"]) for e in entries
    )
    has_old_content = any(
        len(e["old_content"]) > 50 for e in entries
    )

    if readme_changed < MIN_README_CHANGED_LINES:
        continue   # skip PRs with only trivial README changes

    # --- Code patch quality ---
    code_patches = row.get("patches", [])
    code_changed = sum(
        count_meaningful_lines(str(p)) for p in code_patches[:5]
    )

    # --- Commit messages ---
    messages = [m for m in row.get("messages", []) if m and len(m.strip()) > 5]
    has_message = len(messages) > 0

    # --- Score (higher = better golden set candidate) ---
    score = (
        readme_changed * 3        # weight README changes heavily
        + min(code_changed, 30)   # cap code contribution
        + (10 if has_old_content else 0)
        + (5  if has_message else 0)
    )

    scored.append({
        "ghid":              ghid,
        "repo":              row.get("repo", ""),
        "lang":              row.get("lang", ""),
        "score":             score,
        "readme_changed_lines": readme_changed,
        "code_changed_lines":   min(code_changed, 999),
        "has_old_content":   has_old_content,
        "has_message":       has_message,
        "messages":          messages[:3],
        "readme_files":      list(row.get("readme_files", [])),
        "readme_patches":    entries,          # [{file, patch, old_content}]
        "code_patches":      [str(p)[:1500] for p in code_patches[:3]],
    })

scored.sort(key=lambda x: x["score"], reverse=True)

print(f"Scoreable PRs (with meaningful README patch): {len(scored):,}")

# ============================================================
# STEP 4 — show top candidates
# ============================================================

print("\n" + "=" * 60)
print(f"TOP {N_GOLDEN} CANDIDATES")
print("=" * 60)

golden = scored[:N_GOLDEN]

for i, pr in enumerate(golden):
    print(f"\n[{i+1:02d}] {pr['repo']}  |  lang={pr['lang']}  |  score={pr['score']}")
    print(f"     README changed lines : {pr['readme_changed_lines']}")
    print(f"     Code changed lines   : {pr['code_changed_lines']}")
    print(f"     Has old README       : {pr['has_old_content']}")
    print(f"     Commit message       : {pr['messages'][0][:80] if pr['messages'] else '(empty)'}")

    # Show README patch preview
    for entry in pr["readme_patches"][:1]:
        patch_preview = entry["patch"][:300].replace("\n", "\n     ")
        print(f"\n     README patch preview:\n     {patch_preview}")

# ============================================================
# STEP 5 — save
# ============================================================

print("\n" + "=" * 60)
print("STEP 5: Saving golden set")
print("=" * 60)

OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)

# Save as JSON (easy to inspect manually)
with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(golden, f, indent=2, ensure_ascii=False)

print(f"JSON  saved: {OUTPUT_JSON}")

# Save as parquet (drop nested lists for compatibility)
parquet_rows = []
for pr in golden:
    parquet_rows.append({
        "ghid":                  pr["ghid"],
        "repo":                  pr["repo"],
        "lang":                  pr["lang"],
        "score":                 pr["score"],
        "readme_changed_lines":  pr["readme_changed_lines"],
        "code_changed_lines":    pr["code_changed_lines"],
        "has_old_content":       pr["has_old_content"],
        "has_message":           pr["has_message"],
        "messages":              pr["messages"],
        "readme_files":          pr["readme_files"],
        # Store complex fields as JSON strings
        "readme_patches_json":   json.dumps(pr["readme_patches"], ensure_ascii=False),
        "code_patches_json":     json.dumps(pr["code_patches"],   ensure_ascii=False),
    })

pd.DataFrame(parquet_rows).to_parquet(OUTPUT_PARQUET, index=False)
print(f"Parquet saved: {OUTPUT_PARQUET}")

print(f"\nDone. Golden set contains {len(golden)} PRs.")
print("\nNext step: run generate_docs_multi_llm.py")
