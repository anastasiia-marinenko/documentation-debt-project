"""
scripts/llm_generation/generate_docs.py

Generates README update suggestions using Claude API.
Input:  data/processed/readme_prs.parquet
Output: data/results/llm_generated.jsonl
"""

import json
import time
from pathlib import Path

import pandas as pd
import requests

# ============================================================
# CONFIG — edit these
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[2]

INPUT_FILE  = ROOT_DIR / "data/processed/readme_prs.parquet"
OUTPUT_FILE = ROOT_DIR / "data/results/llm_generated.jsonl"

MODEL       = "claude-sonnet-4-20250514"
MAX_TOKENS  = 1024
N_SAMPLES   = 5          # how many PRs to process (start small)
SLEEP_SEC   = 1.0        # pause between API calls

# ============================================================
# LOAD
# ============================================================

print("=" * 60)
print("Loading README PR dataset...")
df = pd.read_parquet(INPUT_FILE)
print(f"Loaded: {len(df):,} PRs")

df_sample = df.head(N_SAMPLES)

# ============================================================
# PROMPT BUILDER
# ============================================================

def build_prompt(row) -> str:
    repo     = row["repo"]
    lang     = row["lang"]
    messages = "\n".join(f"- {m}" for m in row["messages"][:5])

    # Take first 3 code patches, truncated
    patches = row.get("patches", [])
    code_context = "\n\n".join(str(p)[:800] for p in patches[:3])

    readme_files = "\n".join(f"- {f}" for f in row.get("readme_files", []))

    return f"""You are a software documentation expert.

A developer submitted a pull request to the repository: {repo}
Programming language: {lang}

The following README files were modified:
{readme_files}

Code review comments from the PR:
{messages}

Code changes (diff):
{code_context}

Task:
1. Identify what documentation change is needed based on the code changes above.
2. Write a concrete README update (1-3 sentences) that should be added or changed.
3. Explain in one sentence WHY this update is needed (documentation debt signal).

Format your response as:
DOCUMENTATION UPDATE:
<the actual text to add/change in the README>

REASON:
<why this update is needed>
"""

# ============================================================
# API CALL
# ============================================================

def call_claude(prompt: str) -> str:
    """Call Anthropic API. Returns generated text."""
    url = "https://api.anthropic.com/v1/messages"
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": MODEL,
        "max_tokens": MAX_TOKENS,
        "messages": [{"role": "user", "content": prompt}]
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    # Extract text from response
    return data["content"][0]["text"]

# ============================================================
# MAIN LOOP
# ============================================================

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

results = []
errors  = []

print(f"\nGenerating documentation for {N_SAMPLES} PRs...")
print("=" * 60)

for i, (_, row) in enumerate(df_sample.iterrows()):

    print(f"\n[{i+1}/{N_SAMPLES}] {row['repo']}")

    prompt = build_prompt(row)

    try:
        generated = call_claude(prompt)

        result = {
            "id":          i,
            "repo":        row["repo"],
            "lang":        row["lang"],
            "readme_files": row.get("readme_files", []),
            "messages":    list(row.get("messages", [])),
            "prompt":      prompt,
            "generated":   generated,
            "status":      "ok"
        }

        print(f"  ✓ Generated ({len(generated)} chars)")
        print(f"  Preview: {generated[:200]}...")

    except Exception as e:
        print(f"  ✗ Error: {e}")
        result = {
            "id":     i,
            "repo":   row["repo"],
            "status": "error",
            "error":  str(e)
        }
        errors.append(i)

    results.append(result)

    # Save after each call (in case of crash)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    time.sleep(SLEEP_SEC)

# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("GENERATION COMPLETE")
print("=" * 60)
print(f"Total processed: {len(results)}")
print(f"Successful:      {len(results) - len(errors)}")
print(f"Errors:          {len(errors)}")
print(f"Output saved to: {OUTPUT_FILE}")

# Print all generated outputs
print("\n" + "=" * 60)
print("ALL GENERATED OUTPUTS")
print("=" * 60)

for r in results:
    if r["status"] == "ok":
        print(f"\nRepo: {r['repo']}")
        print(f"Generated:\n{r['generated']}")
        print("-" * 40)