"""
scripts/llm_generation/exp_prompt_variants.py

EXP-003 — Prompt Engineering Comparison
=========================================
Compares five prompt variants (v1–v5) on the same 25 Java documentation pairs.
Only the prompt changes — model, temperature, and dataset are fixed.

Prompt variants:
  v1_minimal      : bare-minimum prompt (lower bound)
  v2_structured   : zero-shot with output format instruction (current baseline)
  v3_template     : Template Pattern — explicit rules + output constraints
  v4_persona      : Persona + Template Pattern — senior engineer persona
  v5_cognitive    : Cognitive Verifier — forced step-by-step reasoning

Research questions answered:
  - Does adding structure improve over a minimal prompt? (v1 → v2)
  - Does a strict output template reduce verbosity/hallucination? (v2 → v3)
  - Does a persona improve accuracy? (v3 → v4)
  - Does step-by-step reasoning help? (v2 → v5)
  - Does meta-prompted v4_meta beat manually designed prompts? (v3/v4 vs v4_meta)

Pattern references:
  Template Pattern, Persona Pattern, Cognitive Verifier — White et al. (2023)
  Meta-Prompting — Suzgun & Kalai (2024)

Configuration:
  Model      : llama-3.1-8b-instant (Groq)   — fixed across all variants
  max_tokens : 512
  temperature: 0.3
  Dataset    : data/processed/doc_pairs_sample.json  (25 Java pairs)
  Output     : data/results/prompt_variants_results.jsonl
               data/results/prompt_variants_metrics.csv

Usage:
    python scripts/llm_generation/exp_prompt_variants.py

Requires:
    .env with GROQ_API_KEY
    data/processed/doc_pairs_sample.json  (from extract_doc_pairs.py)
    pip install groq rouge-score bert-score
"""

import json
import os
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

# ============================================================
# CONFIG
# ============================================================

ROOT       = Path(__file__).resolve().parents[2]
INPUT_JSON = ROOT / "data/processed/doc_pairs_sample.json"
OUT_JSONL  = ROOT / "data/results/prompt_variants_results.jsonl"
OUT_CSV    = ROOT / "data/results/prompt_variants_metrics.csv"

MODEL       = "llama-3.1-8b-instant"
MAX_TOKENS  = 512
TEMPERATURE = 0.3
SLEEP_SEC   = 0.4   # between Groq calls

load_dotenv(ROOT / ".env")
GROQ_KEY = os.getenv("GROQ_API_KEY", "")

print("=" * 60)
print("EXP-003 — Prompt Engineering Comparison")
print("=" * 60)
print(f"Model     : {MODEL}")
print(f"Groq key  : {'✓ set' if GROQ_KEY else '✗ missing — set GROQ_API_KEY in .env'}")

# ============================================================
# PROMPT VARIANTS
# ============================================================
# Each variant isolates one design decision from the literature.
# {code} is replaced with the actual method before each call.

PROMPTS = {

    # ── v1: absolute minimum ────────────────────────────────
    # No instructions, no format, just the task.
    # Establishes the lower bound — everything above is prompt engineering gain.
    "v1_minimal": (
        "Write a Javadoc comment for this Java method.\n\n"
        "{code}"
    ),

    # ── v2: structured zero-shot ─────────────────────────────
    # Standard zero-shot prompt with output format instruction.
    # This is the current baseline from exp_doc_pairs_llm.py.
    "v2_structured": (
        "You are a Java documentation expert.\n\n"
        "Write a complete Javadoc comment for the following Java method.\n"
        "Include:\n"
        "- A one-sentence summary describing what the method does\n"
        "- @param tag for each parameter\n"
        "- @return tag if the method returns a value\n"
        "- @throws tag if the method may throw exceptions\n\n"
        "Output ONLY the Javadoc block (/** ... */). No explanation, no code.\n\n"
        "```java\n{code}\n```"
    ),

    # ── v3: Template Pattern (White et al., 2023) ────────────
    # Strict rules + explicit output constraints.
    # Template Pattern: defines exactly what the output must contain.
    # Hypothesis: reduces verbosity and scope creep vs v2.
    "v3_template": (
        "You are a Java documentation expert.\n\n"
        "Write a Javadoc comment for the method below.\n\n"
        "Rules — follow exactly:\n"
        "- Summary line: one sentence only, describe what the method does\n"
        "- @param: one tag per parameter, include the parameter name and a brief description\n"
        "- @return: one line describing the return value; omit entirely if the method is void\n"
        "- @throws: one tag per exception type actually declared or thrown; omit if none\n"
        "- Do NOT add tags that are not supported by the method signature\n"
        "- Output ONLY the Javadoc block (/** ... */) — no prose, no code, no explanation\n\n"
        "```java\n{code}\n```"
    ),

    # ── v4: Persona + Template Pattern (White et al., 2023) ──
    # Assigns an expert role before giving structured instructions.
    # Persona Pattern: the role primes the model for precision.
    # Hypothesis: persona + template together reduce hallucination vs v3 alone.
    "v4_persona": (
        "Act as a senior Java engineer writing public API documentation "
        "for a production codebase. Your documentation will be read by "
        "other engineers integrating this code. Precision matters.\n\n"
        "For the method below, write a Javadoc comment that:\n"
        "- Opens with one concise sentence summarising the method's purpose\n"
        "- Has a @param tag for every parameter (name + what it represents)\n"
        "- Has a @return tag describing what is returned (omit if void)\n"
        "- Has a @throws tag for each checked exception (omit if none)\n"
        "- Uses at most two sentences per tag\n"
        "- Contains NO information that is not present in the method\n\n"
        "Output ONLY the Javadoc block (/** ... */). No prose, no explanation.\n\n"
        "```java\n{code}\n```"
    ),

    # ── v5: Cognitive Verifier (White et al., 2023) ──────────
    # Forces the model to reason step-by-step before generating.
    # Cognitive Verifier: sub-questions reduce hallucination on
    # ambiguous inputs (e.g. methods with sparse or misleading names).
    # Hypothesis: step-by-step grounding improves accuracy vs v2 baseline.
    "v5_cognitive": (
        "You are a Java documentation expert.\n\n"
        "Before writing the Javadoc comment, read the method carefully and "
        "answer these questions to yourself:\n"
        "  1. What does this method do? (one sentence)\n"
        "  2. What does each parameter represent?\n"
        "  3. What does it return, if anything?\n"
        "  4. Does it throw or declare any exceptions?\n\n"
        "Now write a Javadoc comment based ONLY on your answers above.\n"
        "Do not add information that is not present in the method.\n\n"
        "Output ONLY the Javadoc block (/** ... */).\n\n"
        "```java\n{code}\n```"
    ),
}

# ============================================================
# GROQ CALLER
# ============================================================

def call_groq(prompt: str) -> str:
    if not GROQ_KEY:
        return "SKIPPED: GROQ_API_KEY not set"
    try:
        from groq import Groq
        client = Groq(api_key=GROQ_KEY)
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
        )
        return resp.choices[0].message.content.strip()
    except ImportError:
        return "ERROR: run: pip install groq"
    except Exception as e:
        return f"ERROR (Groq): {e}"

# ============================================================
# LOAD DATA
# ============================================================

print("\n" + "=" * 60)
print("Loading dataset...")
print("=" * 60)

if not INPUT_JSON.exists():
    print(f"ERROR: {INPUT_JSON} not found.")
    print("Run scripts/preprocessing/extract_doc_pairs.py first.")
    exit(1)

with open(INPUT_JSON, encoding="utf-8") as f:
    pairs = json.load(f)

print(f"Loaded {len(pairs)} Java documentation pairs")
print(f"Prompt variants: {len(PROMPTS)}")
print(f"Total API calls: {len(pairs) * len(PROMPTS)}")
print(f"Estimated time : ~{len(pairs) * len(PROMPTS) * SLEEP_SEC / 60:.1f} min\n")

# ============================================================
# GENERATION LOOP
# ============================================================

results = []
OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)

for variant_name, prompt_template in PROMPTS.items():
    print(f"\n{'='*60}")
    print(f"Variant: {variant_name}")
    print(f"{'='*60}")

    for i, pair in enumerate(pairs):
        code = pair["method_code"][:1500]   # truncate to avoid token overflow
        gt   = pair["javadoc_gt"]
        repo = pair.get("repo", "unknown")

        prompt = prompt_template.replace("{code}", code)

        print(f"  [{i+1:02d}/{len(pairs)}] {repo[:45]}", end=" ", flush=True)
        t0 = time.time()
        generated = call_groq(prompt)
        elapsed   = time.time() - t0

        is_error = generated.startswith(("ERROR", "SKIPPED"))
        print(f"{'✗' if is_error else '✓'} ({elapsed:.1f}s)")
        if is_error:
            print(f"           {generated[:80]}")

        results.append({
            "variant":      variant_name,
            "pair_id":      i,
            "repo":         repo,
            "method_code":  code,
            "ground_truth": gt,
            "prompt":       prompt,
            "generated":    generated,
            "status":       "error" if is_error else "ok",
        })

        # crash-safe save after every row
        with open(OUT_JSONL, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        time.sleep(SLEEP_SEC)

print(f"\nGeneration complete. Saved: {OUT_JSONL}")

# ============================================================
# METRICS — ROUGE + BERTScore
# ============================================================

print("\n" + "=" * 60)
print("Computing metrics (ROUGE-1, ROUGE-L, BERTScore F1)...")
print("=" * 60)

try:
    from rouge_score import rouge_scorer
    from bert_score import score as bscore
except ImportError:
    print("ERROR: pip install rouge-score bert-score")
    exit(1)

sc  = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
df  = pd.DataFrame(results)
ok  = df[df["status"] == "ok"].copy()

if len(ok) == 0:
    print("No successful outputs to evaluate. Check API key and retry.")
    exit(1)

# ROUGE (row-by-row, fast)
ok["rouge1"] = ok.apply(
    lambda r: round(sc.score(r["ground_truth"], r["generated"])["rouge1"].fmeasure, 4),
    axis=1,
)
ok["rougeL"] = ok.apply(
    lambda r: round(sc.score(r["ground_truth"], r["generated"])["rougeL"].fmeasure, 4),
    axis=1,
)

# BERTScore (batch per variant to save memory)
bertscore_vals = []
for variant in ok["variant"].unique():
    vdf = ok[ok["variant"] == variant]
    print(f"  BERTScore: {variant} ({len(vdf)} samples)...", end=" ", flush=True)
    _, _, F1 = bscore(
        vdf["generated"].tolist(),
        vdf["ground_truth"].tolist(),
        lang="en",
        verbose=False,
    )
    vals = [round(x.item(), 4) for x in F1]
    bertscore_vals.extend(zip(vdf.index, vals))
    print("done")

bertscore_map = dict(bertscore_vals)
ok["bertscore"] = ok.index.map(bertscore_map)

# Save full results
ok.drop(columns=["prompt", "method_code"], errors="ignore").to_csv(OUT_CSV, index=False)
print(f"\nMetrics saved: {OUT_CSV}")

# ============================================================
# SUMMARY TABLE
# ============================================================

print("\n" + "=" * 60)
print("RESULTS — Prompt Variants v1–v5")
print("Fixed model: llama-3.1-8b-instant (Groq)")
print("=" * 60)

summary = (
    ok.groupby("variant")[["rouge1", "rougeL", "bertscore"]]
    .mean()
    .round(4)
    .rename(columns={"rouge1": "ROUGE-1", "rougeL": "ROUGE-L", "bertscore": "BERTScore F1"})
)

# show in variant order (v1 → v5)
order = [v for v in PROMPTS.keys() if v in summary.index]
summary = summary.loc[order]

print(f"\n{'Variant':<20} {'ROUGE-1':>10} {'ROUGE-L':>10} {'BERTScore':>12}")
print("-" * 55)
for variant, row in summary.iterrows():
    print(f"{variant:<20} {row['ROUGE-1']:>10.4f} {row['ROUGE-L']:>10.4f} {row['BERTScore F1']:>12.4f}")
print("=" * 60)

# delta vs v2 baseline
if "v2_structured" in summary.index:
    print("\nDelta vs v2_structured baseline:")
    base = summary.loc["v2_structured"]
    for variant, row in summary.iterrows():
        if variant == "v2_structured":
            continue
        d1 = row["ROUGE-1"]   - base["ROUGE-1"]
        dB = row["BERTScore F1"] - base["BERTScore F1"]
        print(f"  {variant:<20} ROUGE-1 {d1:+.4f}   BERTScore {dB:+.4f}")

print(f"\nFull results: {OUT_CSV}")
print("\nNext step: run exp_prompt_variants_plot.py to generate figures")
