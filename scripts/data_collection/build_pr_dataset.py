from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
from pathlib import Path

import pandas as pd
import json

# ============================================
# CONFIG
# ============================================

ROOT_DIR = Path(__file__).resolve().parents[2]

INPUT_FILES = [
    ROOT_DIR / "data/raw/train_complete.jsonl",
    ROOT_DIR / "data/raw/valid_complete.jsonl",
    ROOT_DIR / "data/raw/test_complete.jsonl",
]

OUTPUT_FILE = ROOT_DIR / "data/processed/pr_dataset.parquet"

# Common source code extensions
CODE_EXTENSIONS = (
    ".py", ".js", ".ts", ".java", ".cpp",
    ".c", ".cc", ".h", ".hpp",
    ".go", ".rs", ".rb", ".php",
    ".cs", ".swift", ".kt", ".scala"
)

# ============================================
# LOAD DATASET (STREAMING)
# ============================================

print("=" * 60)
print("Loading dataset (streaming mode)...")
print("=" * 60)

grouped = defaultdict(list)

for INPUT_FILE in INPUT_FILES:
    print(f"Reading {INPUT_FILE.name}...")
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line in tqdm(f):

            try:
                row = json.loads(line)

            except json.JSONDecodeError:
                continue

            ghid = row.get("ghid")

            if ghid is None:
                continue

            grouped[ghid].append({

                "repo": row.get("repo"),
                "file": row.get("file"),
                "patch": row.get("patch"),
                "msg": row.get("msg"),
                "lang": row.get("lang"),
                "created_at": row.get("created_at")

            })

print(f"\nUnique PRs found: {len(grouped):,}")

# ============================================
# BUILD PR-LEVEL DATASET
# ============================================

print("\n" + "=" * 60)
print("Building PR-level dataset...")
print("=" * 60)

pr_records = []

for pr_id, changes in tqdm(grouped.items()):

    # ----------------------------------------
    # Extract file paths
    # ----------------------------------------

    files = [

        str(change["file"])

        for change in changes

        if change.get("file") is not None
    ]

    if len(files) == 0:
        continue

    # ----------------------------------------
    # Detect README updates
    # ----------------------------------------

    DOC_EXTENSIONS = (".md", ".rst", ".txt")

    DOCUMENTATION_KEYWORDS = (
        "readme",
        "docs",
        "documentation",
        "wiki",
        "usage",
        "guide",
        "tutorial",
    )

    readme_files = []

    for f in files:

        path_lower = str(f).lower()

        suffix = Path(f).suffix.lower()

        filename = Path(f).name.lower()

        is_doc_extension = suffix in DOC_EXTENSIONS

        has_doc_keyword = any(
            keyword in path_lower
            for keyword in DOCUMENTATION_KEYWORDS
        )

        # exclude source code disguised as docs
        is_source_code = suffix in CODE_EXTENSIONS

        if (
                is_doc_extension
                and has_doc_keyword
                and not is_source_code
        ):
            readme_files.append(f)

    documentation_files = [

        f for f in files

        if Path(f).suffix.lower() in (
            ".md",
            ".rst",
            ".txt"
        )
    ]

    has_readme = len(documentation_files) > 0

    # ----------------------------------------
    # Detect source code changes
    # ----------------------------------------

    code_files = [

        f for f in files

        if f.lower().endswith(CODE_EXTENSIONS)
    ]

    has_code = len(code_files) > 0

    # ----------------------------------------
    # Keep only PRs with:
    # README + code changes
    # ----------------------------------------

    if not (has_readme and has_code):
        continue

    # ----------------------------------------
    # Extract metadata
    # ----------------------------------------

    repo = changes[0].get("repo")
    lang = changes[0].get("lang")
    created_at = changes[0].get("created_at")

    # ----------------------------------------
    # Collect commit messages
    # ----------------------------------------

    messages = []

    for change in changes:

        msg = change.get("msg")

        if msg and msg not in messages:
            messages.append(str(msg))

    # ----------------------------------------
    # Collect patches
    # ----------------------------------------

    patches = []

    for change in changes:

        patch = change.get("patch")

        if patch:
            patches.append(str(patch))

    # ----------------------------------------
    # Build PR sample
    # ----------------------------------------

    pr_record = {

        # identifiers
        "ghid": pr_id,
        "repo": repo,
        "lang": lang,

        # files
        "files": files,
        "readme_files": documentation_files,
        "code_files": code_files,

        # counts
        "num_files": len(files),
        "num_readme_files": len(readme_files),
        "num_code_files": len(code_files),

        # labels
        "has_readme": has_readme,
        "has_code": has_code,

        # textual data
        "messages": messages,
        "patches": patches,

        # metadata
        "created_at": created_at
    }

    pr_records.append(pr_record)

# ============================================
# CREATE FINAL DATAFRAME
# ============================================

print("\n" + "=" * 60)
print("Creating final dataframe...")
print("=" * 60)

pr_df = pd.DataFrame(pr_records)

# ============================================
# FINAL STATISTICS
# ============================================

print("\n" + "=" * 60)
print("FINAL DATASET STATISTICS")
print("=" * 60)

print(f"Final PR-level samples: {len(pr_df):,}")

if len(pr_df) > 0:

    print("\nTop languages:")
    print(pr_df["lang"].value_counts().head(10))

    print("\nTop repositories:")
    print(pr_df["repo"].value_counts().head(10))

    print("\nAverage files per PR:")
    print(round(pr_df["num_files"].mean(), 2))

    print("\nAverage code files per PR:")
    print(round(pr_df["num_code_files"].mean(), 2))

# ============================================
# SAVE DATASET
# ============================================

print("\n" + "=" * 60)
print("Saving dataset...")
print("=" * 60)

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

pr_df.to_parquet(OUTPUT_FILE, index=False)

print(f"Dataset saved to:\n{OUTPUT_FILE}")

# ============================================
# SHOW EXAMPLE
# ============================================

if len(pr_df) > 0:

    print("\n" + "=" * 60)
    print("EXAMPLE PR")
    print("=" * 60)

    sample = pr_df.iloc[0]

    print(f"Repository: {sample['repo']}")
    print(f"Language: {sample['lang']}")
    print(f"Files changed: {sample['num_files']}")

    print("\nREADME files:")

    for f in sample["readme_files"]:
        print("-", f)

    print("\nCode files:")

    for f in sample["code_files"][:10]:
        print("-", f)

    print("\nSample commit messages:")

    for msg in sample["messages"][:3]:
        print("-", msg)

print("\n" + "=" * 60)
print("Done.")
print("=" * 60)