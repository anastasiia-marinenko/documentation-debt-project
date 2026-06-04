"""
scripts/preprocessing/extract_doc_pairs.py

Extracts method-level documentation pairs from Lin et al. (2026)
Code Review dataset — a novel use of code review history for
documentation generation research.

Logic:
  For each Java file row in train_complete.jsonl:
    1. Check if the patch ADDS a Javadoc block (/** ... */ in + lines)
    2. Extract the added Javadoc as ground truth
    3. Find the method body that follows in oldf (old file content)
    4. Build pair: (method_code_without_doc, ground_truth_javadoc)

Why this matters:
  Unlike CodeSearchNet (static snapshots), these are REAL documentation
  updates developers made — directly tied to code changes.
  This is a documentation debt repayment dataset.

Output: data/processed/doc_pairs.parquet
        data/processed/doc_pairs_sample.json  (top N for experiments)
"""

import json
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# ── CONFIG ────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parents[2]
INPUT_FILE = ROOT / "data/raw/train_complete.jsonl"
OUT_PARQUET= ROOT / "data/processed/doc_pairs.parquet"
OUT_JSON   = ROOT / "data/processed/doc_pairs_sample.json"

TARGET_LANG      = "java"          # start with Java (Javadoc)
MIN_JAVADOC_LINES = 3              # at least /** + 1 content line + */
MIN_CODE_LINES    = 4              # method must have some body
N_SAMPLE          = 25             # final golden sample size
# ─────────────────────────────────────────────────────────────────────


def extract_added_javadoc(patch: str) -> list[str]:
    """
    Parse unified diff and extract complete added Javadoc blocks.
    Returns list of Javadoc strings (may be multiple per patch).
    """
    blocks = []
    current = []
    in_block = False

    for raw_line in patch.split("\n"):
        # skip diff headers
        if raw_line.startswith("+++") or raw_line.startswith("---"):
            continue

        is_added   = raw_line.startswith("+")
        is_context = raw_line.startswith(" ")
        content    = raw_line[1:] if (is_added or is_context) else ""
        stripped   = content.strip()

        if is_added:
            if stripped.startswith("/**"):
                in_block = True
                current  = [content]
            elif in_block and (stripped.startswith("*") or stripped == "*/"):
                current.append(content)
                if stripped == "*/" or stripped.endswith("*/"):
                    if len(current) >= MIN_JAVADOC_LINES:
                        blocks.append("\n".join(current))
                    current  = []
                    in_block = False
        else:
            # non-added line interrupts the block
            if in_block:
                in_block = False
                current  = []

    return blocks


def extract_method_from_old_file(oldf: str, hunk_start: int) -> str:
    """
    Given the old file content and approximate line where the Javadoc
    was added, find the Java method body that follows.
    Returns the method code (without any existing comment).
    """
    if not oldf:
        return ""

    lines = oldf.split("\n")
    # search up to 10 lines after hunk_start for a method signature
    search_start = max(0, hunk_start - 5)
    search_end   = min(len(lines), hunk_start + 30)

    sig_pattern = re.compile(
        r"^\s*(public|private|protected|static|final|synchronized|abstract|default)"
        r".*\(.*\)\s*(\{|throws)"
    )

    method_start = None
    for i in range(search_start, search_end):
        if sig_pattern.match(lines[i]):
            method_start = i
            break

    if method_start is None:
        return ""

    # collect method body (simple brace counting)
    depth   = 0
    body    = []
    started = False
    for line in lines[method_start:method_start + 80]:
        body.append(line)
        depth += line.count("{") - line.count("}")
        if "{" in line:
            started = True
        if started and depth <= 0:
            break

    code = "\n".join(body).strip()
    # remove any inline /* */ comments that were already there
    code = re.sub(r"/\*\*.*?\*/", "", code, flags=re.DOTALL).strip()
    return code


def hunk_start_line(patch: str) -> int:
    """Extract the starting line of the first @@ hunk."""
    m = re.search(r"@@ -(\d+)", patch)
    return int(m.group(1)) if m else 0


# ── MAIN EXTRACTION ───────────────────────────────────────────────────
print("=" * 60)
print("Extracting Java documentation pairs from Lin et al. dataset")
print("=" * 60)

pairs = []
skipped_no_patch = 0
skipped_no_javadoc = 0
skipped_no_method = 0

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    for line in tqdm(f, desc="Scanning JSONL"):
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue

        # filter: Java files only
        file_path = str(row.get("file", ""))
        if not file_path.lower().endswith(".java"):
            continue

        patch = str(row.get("patch", "") or "")
        oldf  = str(row.get("oldf",  "") or "")

        if not patch:
            skipped_no_patch += 1
            continue

        # extract added Javadoc blocks
        javadocs = extract_added_javadoc(patch)
        if not javadocs:
            skipped_no_javadoc += 1
            continue

        # find method code in old file
        hunk_line   = hunk_start_line(patch)
        method_code = extract_method_from_old_file(oldf, hunk_line)

        if len(method_code.split("\n")) < MIN_CODE_LINES:
            skipped_no_method += 1
            continue

        for javadoc in javadocs:
            # quality score: longer Javadoc + more tags = better
            tag_count  = sum(1 for t in ["@param","@return","@throws","@see"]
                             if t in javadoc)
            line_count = javadoc.count("\n") + 1
            score      = line_count * 2 + tag_count * 5

            pairs.append({
                "ghid":         str(row.get("ghid", "")),
                "repo":         str(row.get("repo", "")),
                "lang":         str(row.get("lang", "")),
                "file":         file_path,
                "method_code":  method_code,
                "javadoc_gt":   javadoc,
                "tag_count":    tag_count,
                "javadoc_lines":line_count,
                "score":        score,
                "msg":          str(row.get("msg", "") or ""),
            })

# ── STATS ─────────────────────────────────────────────────────────────
print(f"\nExtraction complete")
print(f"  Java rows processed : found in stream")
print(f"  Skipped (no patch)  : {skipped_no_patch:,}")
print(f"  Skipped (no Javadoc): {skipped_no_javadoc:,}")
print(f"  Skipped (no method) : {skipped_no_method:,}")
print(f"  Valid pairs found   : {len(pairs):,}")

if not pairs:
    print("\nNo pairs found. Check that train_complete.jsonl is present.")
    exit(1)

# ── SAVE FULL SET ─────────────────────────────────────────────────────
df = pd.DataFrame(pairs)
df = df.sort_values("score", ascending=False).reset_index(drop=True)
OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
df.to_parquet(OUT_PARQUET, index=False)
print(f"\nFull dataset saved: {OUT_PARQUET}")

# ── TOP-N SAMPLE ──────────────────────────────────────────────────────
sample = df.head(N_SAMPLE).to_dict(orient="records")
with open(OUT_JSON, "w", encoding="utf-8") as f:
    json.dump(sample, f, indent=2, ensure_ascii=False)
print(f"Sample ({N_SAMPLE} pairs) saved: {OUT_JSON}")

# ── PREVIEW ───────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("TOP 3 PAIRS PREVIEW")
print('='*60)
for i, p in enumerate(sample[:3]):
    print(f"\n[{i+1}] {p['repo']}  |  score={p['score']}"
          f"  |  tags={p['tag_count']}  |  javadoc_lines={p['javadoc_lines']}")
    print(f"  Method (first 2 lines):")
    for ln in p['method_code'].split('\n')[:2]:
        print(f"    {ln}")
    print(f"  Ground truth Javadoc (first 2 lines):")
    for ln in p['javadoc_gt'].split('\n')[:2]:
        print(f"    {ln}")

print(f"\nNext step: run exp_doc_pairs_llm.py to generate & evaluate")
