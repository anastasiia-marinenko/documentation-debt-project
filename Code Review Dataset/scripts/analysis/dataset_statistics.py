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

print("=" * 60)
print("Loading README PR dataset...")
print("=" * 60)

df = pd.read_parquet(INPUT_FILE)

print(f"Loaded PRs: {len(df):,}")

# ============================================
# BASIC STATISTICS
# ============================================

print("\n" + "=" * 60)
print("DATASET STATISTICS")
print("=" * 60)

print(f"Total README PRs: {len(df):,}")

print("\nTop languages:")
print(df["lang"].value_counts().head(10))

print("\nTop repositories:")
print(df["repo"].value_counts().head(10))

print("\nAverage files per PR:")
print(round(df["num_files"].mean(), 2))

print("\nAverage code files per PR:")
print(round(df["num_code_files"].mean(), 2))

# ============================================
# README FILE ANALYSIS
# ============================================

all_readme_files = []

for files in df["readme_files"]:
    all_readme_files.extend(files)

readme_series = pd.Series(all_readme_files)

print("\nMost common README paths:")
print(readme_series.value_counts().head(20))

# ============================================
# EXAMPLE PR
# ============================================

if len(df) > 0:

    sample = df.iloc[0]

    print("\n" + "=" * 60)
    print("EXAMPLE PR")
    print("=" * 60)

    print(f"Repository: {sample['repo']}")
    print(f"Language: {sample['lang']}")

    print("\nREADME files:")
    for f in sample["readme_files"]:
        print("-", f)

    print("\nMessages:")
    for msg in sample["messages"][:3]:
        print("-", msg)

print("\nDone.")