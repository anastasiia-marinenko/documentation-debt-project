from pathlib import Path
import pandas as pd

# ============================================
# CONFIG
# ============================================

ROOT_DIR = Path(__file__).resolve().parents[2]

INPUT_FILE = ROOT_DIR / "data/processed/readme_prs.parquet"

# ============================================
# LOAD DATASET
# ============================================

df = pd.read_parquet(INPUT_FILE)

print(f"Loaded PRs: {len(df):,}")

# ============================================
# TAKE SAMPLE
# ============================================

sample = df.iloc[0]

print("=" * 60)
print("SAMPLE PR")
print("=" * 60)

print(f"Repository: {sample['repo']}")
print(f"Language: {sample['lang']}")

print("\nREADME files:")
for f in sample["readme_files"]:
    print("-", f)

print("\nCommit messages:")
for msg in sample["messages"]:
    print("-", msg)

print("\nPatch preview:")

first_patch = sample["patches"][0]

print(first_patch[:2000])

# ============================================
# BUILD PROMPT
# ============================================

prompt = f"""
You are a software documentation assistant.

A pull request modified source code and README files.

Repository:
{sample['repo']}

Programming language:
{sample['lang']}

Commit messages:
{sample['messages']}

Code changes:
{first_patch[:4000]}

Task:
Generate a README update summary describing the changes.
"""

print("\n" + "=" * 60)
print("PROMPT")
print("=" * 60)

print(prompt)

print("\nDone.")