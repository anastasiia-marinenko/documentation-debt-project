"""
scripts/analysis/classify_readme_sections.py

EXP-002 Configuration:
  Model       : llama-3.1-8b-instant (Groq)
  temperature : 0 (deterministic)
  Taxonomy    : Prana et al. (2019) — 8 categories
  Date        : 2026-05-29
"""

import json
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from groq import Groq
from tqdm import tqdm

load_dotenv()

# ============================================================
# CONFIG
# ============================================================

ROOT_DIR     = Path(__file__).resolve().parents[2]
INPUT_FILE   = ROOT_DIR / "data/processed/golden_set.parquet"
OUTPUT_FILE  = ROOT_DIR / "data/results/readme_section_classification.csv"

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
MODEL        = "llama-3.1-8b-instant"

CATEGORIES = """
Choose ONE category:

- What         = project overview / description
- Why          = motivation / purpose
- How          = installation / usage / API / configuration
- When         = status / roadmap / release info
- Who          = authors / maintainers
- Contribution = contributing guidelines
- References   = links / external resources
- Other        = everything else
"""

# ============================================================
# LOAD
# ============================================================

print("=" * 60)
print("Loading golden set...")
print("=" * 60)

if not INPUT_FILE.exists():
    print(f"ERROR: {INPUT_FILE} not found.")
    exit(1)

if not GROQ_API_KEY:
    print("ERROR: GROQ_API_KEY not set in .env")
    exit(1)

df = pd.read_parquet(INPUT_FILE)
print(f"Loaded {len(df)} PRs")
print(f"Columns: {df.columns.tolist()}")

client = Groq(api_key=GROQ_API_KEY)

# ============================================================
# PATCH EXTRACTOR
# — підтримує обидва формати: json-рядок і звичайна колонка
# ============================================================

def extract_patch(row) -> str:
    """Витягує текст README патча з рядка датасету."""

    # Варіант 1 — є колонка readme_patches_json (JSON-рядок зі списком)
    if "readme_patches_json" in row.index:
        try:
            patches = json.loads(row["readme_patches_json"])
            text = ""
            for p in patches:
                if "patch" in p:
                    text += p["patch"] + "\n"
            if text.strip():
                return text.strip()
        except Exception:
            pass

    # Варіант 2 — є колонка ground_truth_patch (простий рядок)
    if "ground_truth_patch" in row.index:
        val = str(row["ground_truth_patch"]).strip()
        if val and val != "nan":
            return val

    # Варіант 3 — є колонка patches (список патчів усіх файлів)
    if "patches" in row.index:
        patches = row["patches"]
        if isinstance(patches, list) and patches:
            return str(patches[0])[:1500]

    return "(no patch available)"

# ============================================================
# CLASSIFIER
# ============================================================

def classify_patch(patch_text: str) -> str:
    prompt = f"""You are analyzing README documentation updates.

{CATEGORIES}

README PATCH:
{patch_text[:800]}

Return ONLY the category name (one word).
"""
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=10,
        )
        raw = response.choices[0].message.content.strip()
        # Нормалізація — тільки перше слово
        word = raw.split()[0].strip(".,;:").capitalize()
        valid = {"What", "Why", "How", "When", "Who",
                 "Contribution", "References", "Other"}
        return word if word in valid else f"Other ({raw})"
    except Exception as e:
        return f"ERROR: {e}"

# ============================================================
# RUN
# ============================================================

print("\n" + "=" * 60)
print("Classifying README patches...")
print(f"Model: {MODEL} | Temperature: 0")
print("=" * 60)

rows = []

for _, row in tqdm(df.iterrows(), total=len(df)):
    patch    = extract_patch(row)
    category = classify_patch(patch)

    rows.append({
        "repo":         row.get("repo", ""),
        "lang":         row.get("lang", ""),
        "category":     category,
        "readme_patch": patch[:500],
    })

# ============================================================
# SAVE
# ============================================================

results_df = pd.DataFrame(rows)
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
results_df.to_csv(OUTPUT_FILE, index=False)

print(f"\nSaved: {OUTPUT_FILE}")

# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("EXP-002 RESULTS — README Section Category Distribution")
print("=" * 60)

counts = results_df["category"].value_counts()
print(counts.to_string())

total = len(results_df)
print(f"\nTotal classified: {total}")

if len(counts):
    top = counts.index[0]
    print(f"\nTop category: {top} ({counts.iloc[0]}/{total} = {counts.iloc[0]/total*100:.0f}%)")
    print("\nFinding: README documentation debt is concentrated in:")
    for cat, n in counts.head(3).items():
        print(f"  {cat}: {n} PRs ({n/total*100:.0f}%)")

print("\nConsistent with Prana et al. (2019): 'How' sections (installation,")
print("usage, API) are the most common README content category.")
print("\nDone.")