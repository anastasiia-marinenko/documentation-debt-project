"""
scripts/llm_generation/meta_prompting.py

Meta-prompting experiment: ask an LLM to improve the generation prompt,
then re-run generation with the improved prompt and compare metrics.

Pipeline:
  Step 1 — Meta-prompt: send current prompt (v3) to Groq and ask it to
            improve it for Javadoc generation quality.
  Step 2 — Generation: run the improved prompt (v4) on the same 25 samples.
  Step 3 — Evaluation: compute ROUGE + BERTScore for v3 vs v4.
  Step 4 — Save comparison: data/results/meta_prompt_comparison.csv

Usage:
    python scripts/llm_generation/meta_prompting.py

Requires:
    .env with GROQ_API_KEY
    data/processed/doc_pairs_sample.json  (from extract_doc_pairs.py)
    data/results/exp_doc_pairs_metrics.csv (v3 baseline results)
"""

import json, os, time
from pathlib import Path
from dotenv import load_dotenv

ROOT       = Path(__file__).resolve().parents[2]
INPUT_JSON = ROOT / "data/processed/doc_pairs_sample.json"
V3_RESULTS = ROOT / "data/results/exp_doc_pairs_results.jsonl"
OUT_JSONL  = ROOT / "data/results/meta_prompt_results.jsonl"
OUT_CSV    = ROOT / "data/results/meta_prompt_comparison.csv"
OUT_PROMPT = ROOT / "data/results/improved_prompt_v4.txt"

MAX_TOKENS  = 512
TEMPERATURE = 0.3
MODEL       = "llama-3.1-8b-instant"

load_dotenv(ROOT / ".env")
GROQ_KEY = os.getenv("GROQ_API_KEY", "")

# ── CURRENT PROMPT v3 (baseline) ─────────────────────────────
PROMPT_V3 = """You are a Java documentation expert.

Write a complete Javadoc comment for the following Java method.
Include:
- A one-sentence summary describing what the method does
- @param tag for each parameter
- @return tag if the method returns a value
- @throws tag if the method may throw exceptions

Output ONLY the Javadoc block (/** ... */). No explanation, no code.

```java
{code}
```"""

# ── STEP 1: META-PROMPT ───────────────────────────────────────

META_INSTRUCTION = """You are a prompt engineering expert specialising in code documentation generation.

Below is a prompt currently used to ask an LLM to generate Javadoc documentation for a Java method.
Your task: rewrite this prompt so that the LLM generates MORE ACCURATE, MORE COMPLETE, and
BETTER-STRUCTURED Javadoc comments.

Specifically, try to improve:
1. The instruction clarity (what exactly is expected)
2. The output format specification (reduce hallucination, enforce structure)
3. The quality of @param / @return / @throws coverage
4. Conciseness — the generated Javadoc should be useful, not verbose

Rules:
- Keep the {code} placeholder exactly as is
- Return ONLY the improved prompt text — no explanation, no preamble
- The improved prompt must still be self-contained and work zero-shot

Current prompt:
---
CURRENT_PROMPT_HERE
---

Improved prompt:"""

def call_groq(prompt: str, max_tokens: int = MAX_TOKENS) -> str:
    if not GROQ_KEY:
        raise ValueError("GROQ_API_KEY not set")
    from groq import Groq
    client = Groq(api_key=GROQ_KEY)
    r = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=TEMPERATURE,
    )
    return r.choices[0].message.content.strip()

# ── MAIN ──────────────────────────────────────────────────────

print("=" * 60)
print("STEP 1 — Meta-prompting: improving the generation prompt")
print("=" * 60)

meta_request = META_INSTRUCTION.replace("{current_prompt}", PROMPT_V3)
print("Sending meta-prompt to LLM...")
t0 = time.time()
PROMPT_V4 = call_groq(meta_request, max_tokens=1024)
print(f"Done ({time.time()-t0:.1f}s)")

print("\n--- IMPROVED PROMPT v4 ---")
print(PROMPT_V4)
print("-" * 40)

OUT_PROMPT.parent.mkdir(parents=True, exist_ok=True)
OUT_PROMPT.write_text(PROMPT_V4, encoding="utf-8")
print(f"\nSaved improved prompt: {OUT_PROMPT}")

# ── STEP 2: GENERATION WITH V4 ────────────────────────────────

print("\n" + "=" * 60)
print("STEP 2 — Running generation with improved prompt (v4)")
print("=" * 60)

with open(INPUT_JSON, encoding="utf-8") as f:
    pairs = json.load(f)

print(f"Loaded {len(pairs)} pairs")

results_v4 = []
OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)

for i, pair in enumerate(pairs):
    code = pair["method_code"]
    gt   = pair["javadoc_gt"]

    # fill template
    prompt = PROMPT_V4.replace("{code}", code[:1500])

    print(f"[{i+1:02d}/{len(pairs)}] {pair.get('repo','?')[:40]}", end=" ", flush=True)
    try:
        generated = call_groq(prompt)
        status = "ok"
        print("✓")
    except Exception as e:
        generated = f"ERROR: {e}"
        status = "error"
        print("✗")

    results_v4.append({
        "id":           i,
        "repo":         pair["repo"],
        "method_code":  code,
        "javadoc_gt":   gt,
        "prompt_v4":    prompt,
        "generated_v4": generated,
        "status":       status,
    })

    with open(OUT_JSONL, "w", encoding="utf-8") as f:
        for r in results_v4:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    time.sleep(0.3)

print(f"\nGeneration complete. Saved: {OUT_JSONL}")

# ── STEP 3: METRICS ───────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 3 — Computing ROUGE + BERTScore for v3 and v4")
print("=" * 60)

import pandas as pd
from rouge_score import rouge_scorer
from bert_score import score as bscore

scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)

# Load v3 results
v3_rows = {}
if V3_RESULTS.exists():
    with open(V3_RESULTS, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("status") == "ok":
                v3_rows[r["id"]] = r.get("groq_output", "")

rows = []
for r in results_v4:
    if r["status"] != "ok":
        continue
    gt  = r["javadoc_gt"]
    g4  = r["generated_v4"]
    g3  = v3_rows.get(r["id"], "")

    rs4 = scorer.score(gt, g4)
    rs3 = scorer.score(gt, g3) if g3 else None

    rows.append({
        "id":       r["id"],
        "repo":     r["repo"],
        "gt":       gt,
        "v3":       g3,
        "v4":       g4,
        "v3_rouge1": round(rs3["rouge1"].fmeasure, 4) if rs3 else None,
        "v3_rougeL": round(rs3["rougeL"].fmeasure, 4) if rs3 else None,
        "v4_rouge1": round(rs4["rouge1"].fmeasure, 4),
        "v4_rougeL": round(rs4["rougeL"].fmeasure, 4),
    })

df = pd.DataFrame(rows)
print("Computing BERTScore...")
_, _, F1_v4 = bscore(df["v4"].tolist(), df["gt"].tolist(), lang="en", verbose=False)
df["v4_bertscore"] = [round(x.item(), 4) for x in F1_v4]

if df["v3"].notna().all() and (df["v3"] != "").all():
    _, _, F1_v3 = bscore(df["v3"].tolist(), df["gt"].tolist(), lang="en", verbose=False)
    df["v3_bertscore"] = [round(x.item(), 4) for x in F1_v3]
else:
    df["v3_bertscore"] = None

df.to_csv(OUT_CSV, index=False)

# ── STEP 4: SUMMARY ──────────────────────────────────────────

print("\n" + "=" * 60)
print("RESULTS — Prompt v3 (baseline) vs v4 (meta-improved)")
print("=" * 60)

def mean(col):
    return round(df[col].dropna().mean(), 4)

print(f"{'Metric':<20} {'v3 (baseline)':>15} {'v4 (meta)':>15} {'Delta':>10}")
print("-" * 62)
for m, v3c, v4c in [
    ("ROUGE-1",     "v3_rouge1",    "v4_rouge1"),
    ("ROUGE-L",     "v3_rougeL",    "v4_rougeL"),
    ("BERTScore F1","v3_bertscore", "v4_bertscore"),
]:
    v3m = mean(v3c) if v3c in df else "-"
    v4m = mean(v4c)
    delta = round(v4m - v3m, 4) if isinstance(v3m, float) else "-"
    arrow = ("↑" if isinstance(delta, float) and delta > 0
             else "↓" if isinstance(delta, float) and delta < 0 else "=")
    print(f"{m:<20} {str(v3m):>15} {str(v4m):>15} {str(delta):>8} {arrow}")

print(f"\nFull comparison saved: {OUT_CSV}")
print(f"Improved prompt saved: {OUT_PROMPT}")
