from pathlib import Path
import pandas as pd

# ============================================
# CONFIG
# ============================================

ROOT_DIR = Path(__file__).resolve().parents[2]

INPUT_FILE = ROOT_DIR / "data/processed/pr_dataset.parquet"
OUTPUT_FILE = ROOT_DIR / "data/processed/readme_prs.parquet"

# ============================================
# LOAD DATASET
# ============================================

print("=" * 60)
print("Loading PR dataset...")
print("=" * 60)

df = pd.read_parquet(INPUT_FILE)

print(f"Loaded PRs: {len(df):,}")

# ============================================
# FILTER README PRs
# ============================================

print("\nFiltering README-related PRs...")

readme_df = df[
    (df["has_readme"] == True) &
    (df["has_code"] == True)
].copy()

print(f"README + code PRs: {len(readme_df):,}")

# ============================================
# SAVE
# ============================================

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

readme_df.to_parquet(OUTPUT_FILE, index=False)

print("\nDataset saved:")
print(OUTPUT_FILE)

# ============================================
# EXAMPLES
# ============================================

if len(readme_df) > 0:

    sample = readme_df.iloc[0]

    print("\nExample repository:")
    print(sample["repo"])

    print("\nREADME files:")
    print(sample["readme_files"])

print("\nDone.")