"""
scripts/llm_generation/generate_docs_multi_llm.py

EXP-001 (v2) — Inline Code Documentation Generation
=======================================================
Reframed task: instead of generating a README update, the LLM now generates
INLINE documentation (XML doc comments, Javadoc, docstrings, etc.) for code
changes that lack or have outdated documentation — which is what the Lin et al.
dataset actually contains.

Configuration:
  Models       : llama-3.1-8b-instant (Groq), gemini-2.5-flash (Gemini),
                 meta-llama/Meta-Llama-3-8B-Instruct (HuggingFace)
  max_tokens   : 1024 (Groq/HF), 2048 (Gemini)
  temperature  : 0.3 (Groq/Gemini), 0.2 (HF)
  Prompt ver   : v3 (inline doc generation)
  Input        : code diff + commit messages + old file content
  Output format: INLINE DOCUMENTATION / PLACEMENT / REASON
  Date         : 2026-06-02
"""

import json
import os
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# CONFIG
# ============================================================

ROOT_DIR     = Path(__file__).resolve().parents[2]
INPUT_FILE   = ROOT_DIR / "data/processed/golden_set.parquet"
OUTPUT_JSONL = ROOT_DIR / "data/results/multi_llm_results.jsonl"
OUTPUT_CSV   = ROOT_DIR / "data/results/comparison_table.csv"

MODEL_GROQ   = "llama-3.1-8b-instant"
MODEL_GEMINI = "gemini-2.5-flash"
MODEL_HF     = "meta-llama/Meta-Llama-3-8B-Instruct"

MAX_TOKENS        = 1024   # Groq / HuggingFace
MAX_TOKENS_GEMINI = 2048   # Gemini needs more room for its verbose formatting
TEMPERATURE       = 0.3
SLEEP_BETWEEN     = 4      # seconds between calls (Gemini: 15 RPM free tier)

GROQ_API_KEY   = os.getenv("GROQ_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
HF_TOKEN       = os.getenv("HF_TOKEN", "")

# ============================================================
# STATUS
# ============================================================

print("=" * 60)
print("API KEY STATUS")
print("=" * 60)
print(f"Groq    : {'✓ set' if GROQ_API_KEY   else '✗ missing'}")
print(f"Gemini  : {'✓ set' if GEMINI_API_KEY else '✗ missing'}")
print(f"HF Token: {'✓ set' if HF_TOKEN       else '✗ missing'}")

# ============================================================
# LANGUAGE → DOC STYLE MAPPING
# ============================================================

LANG_DOC_STYLE = {
    ".cs":   "XML doc comments (/// <summary>...</summary>)",
    ".java": "Javadoc (/** ... */)",
    ".py":   'Python docstring ("""...""")',
    ".cpp":  "Doxygen (/** ... */ or ///<)",
    ".c":    "Doxygen or C block comment (/** ... */)",
    ".go":   "Go doc comment (// FunctionName ...)",
    ".js":   "JSDoc (/** @param ... @returns ... */)",
    ".ts":   "TSDoc (/** @param ... @returns ... */)",
    ".rb":   "YARD (# @param ... @return ...)",
    ".rs":   "Rust doc comment (/// ...)",
}

def get_doc_style(lang: str) -> str:
    return LANG_DOC_STYLE.get(lang.lower(), "appropriate inline documentation comment")

# ============================================================
# PROMPT BUILDER  v3 — inline code documentation
# ============================================================

def build_prompt(row: pd.Series) -> str:
    repo  = row.get("repo", "unknown")
    lang  = row.get("lang", "unknown")
    msgs  = list(row.get("messages", []))[:5]

    # Old file content (before PR)
    old_content = ""
    try:
        patches = json.loads(row.get("readme_patches_json", "[]"))
        for p in patches[:1]:
            old_content = str(p.get("old_content", ""))[:2000]
    except Exception:
        pass

    # Code diff
    code_diff = ""
    try:
        code_patches = json.loads(row.get("code_patches_json", "[]"))
        code_diff = "\n\n".join(str(p)[:800] for p in code_patches[:2])
    except Exception:
        pass
    if not code_diff:
        # Fallback: use the patch from readme_patches_json (which is actually code)
        try:
            patches = json.loads(row.get("readme_patches_json", "[]"))
            code_diff = patches[0].get("patch", "") if patches else ""
        except Exception:
            pass

    doc_style = get_doc_style(lang)
    msg_text  = "\n".join(f"- {m}" for m in msgs) if msgs else "(none)"

    return f"""You are an expert software engineer helping eliminate documentation debt.

Repository : {repo}
Language   : {lang}
Doc style  : {doc_style}

Commit messages:
{msg_text}

File content BEFORE this pull request:
---
{old_content if old_content else "(not available)"}
---

Code changes (diff) — lines starting with + are added, - are removed:
---
{code_diff if code_diff else "(not available)"}
---

Task:
The code above was changed but inline documentation is missing or outdated.
Generate the inline documentation comment(s) that should accompany these changes.
Use {doc_style} format. Be concrete and accurate — do NOT invent functionality.

Respond ONLY in this format:

INLINE DOCUMENTATION:
<the exact comment(s) to add, in the correct doc format for {lang}>

PLACEMENT:
<where exactly to place this comment — e.g. "above the NatGateway property" or "replace the copy() docstring">

REASON:
<one sentence: why this documentation is needed / what debt it addresses>"""

# ============================================================
# LLM CALLERS
# ============================================================

def call_groq(prompt: str) -> str:
    if not GROQ_API_KEY:
        return "SKIPPED: GROQ_API_KEY not set"
    try:
        from groq import Groq
        client = Groq(api_key=GROQ_API_KEY)
        resp = client.chat.completions.create(
            model=MODEL_GROQ,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
        )
        return resp.choices[0].message.content.strip()
    except ImportError:
        return "ERROR: run: pip install groq"
    except Exception as e:
        return f"ERROR (Groq): {e}"


def call_gemini(prompt: str) -> str:
    if not GEMINI_API_KEY:
        return "SKIPPED: GEMINI_API_KEY not set"
    try:
        from google import genai
        from google.genai import types
        client = genai.Client(api_key=GEMINI_API_KEY)
        response = client.models.generate_content(
            model=MODEL_GEMINI,
            contents=prompt,
            config=types.GenerateContentConfig(
                max_output_tokens=MAX_TOKENS_GEMINI,
                temperature=TEMPERATURE,
            ),
        )
        return response.text.strip()
    except ImportError:
        return "ERROR: run: pip install google-genai"
    except Exception as e:
        err = str(e)
        if "429" in err:
            return "ERROR (Gemini): 429 quota exceeded — wait 24h or upgrade to paid tier"
        if "404" in err:
            return f"ERROR (Gemini): model not found — check MODEL_GEMINI. {e}"
        return f"ERROR (Gemini): {e}"


def call_huggingface(prompt: str) -> str:
    if not HF_TOKEN:
        return "SKIPPED: HF_TOKEN not set"
    try:
        from huggingface_hub import InferenceClient
        client = InferenceClient(api_key=HF_TOKEN)
        result = client.chat_completion(
            model=MODEL_HF,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=MAX_TOKENS,
            temperature=0.2,
        )
        output = result.choices[0].message.content.strip()
        # Strip Llama boilerplate prefix
        for prefix in ["Here is the response:", "Here is the response :",
                       "Here's the response:", "Here is my response:"]:
            if output.startswith(prefix):
                output = output[len(prefix):].strip()
                break
        if "[/INST]" in output:
            output = output.split("[/INST]")[-1].strip()
        return output
    except ImportError:
        return "ERROR: run: pip install huggingface_hub"
    except Exception as e:
        err = str(e)
        if "403" in err:
            return "ERROR (HF): 403 — token lacks Inference API permission. huggingface.co/settings/tokens"
        if "400" in err:
            return "ERROR (HF): 400 — model does not support this request type"
        return f"ERROR (HuggingFace): {e}"

# ============================================================
# HELPERS
# ============================================================

def clean_output(text: str) -> str:
    """Strip anything before INLINE DOCUMENTATION:"""
    marker = "INLINE DOCUMENTATION:"
    idx = text.find(marker)
    if idx != -1:
        return text[idx:].strip()
    return text.strip()


def get_ground_truth_patch(row) -> str:
    try:
        patches = json.loads(row.get("readme_patches_json", "[]"))
        return patches[0].get("patch", "") if patches else ""
    except Exception:
        return ""

# ============================================================
# LOAD
# ============================================================

print("\n" + "=" * 60)
print("Loading golden set...")
print("=" * 60)

if not INPUT_FILE.exists():
    print(f"ERROR: {INPUT_FILE} not found.")
    print("Run scripts/preprocessing/select_golden_set.py first.")
    exit(1)

df = pd.read_parquet(INPUT_FILE)
print(f"Loaded {len(df)} PRs for generation")
print(f"Languages: {df['lang'].unique().tolist() if 'lang' in df.columns else 'unknown'}")

# ============================================================
# GENERATION LOOP
# ============================================================

CALLERS = {
    "groq":        call_groq,
    "gemini":      call_gemini,
    "huggingface": call_huggingface,
}

results = []

print("\n" + "=" * 60)
print("EXP-001 v2 — Inline Code Documentation Generation")
print(f"Task: generate inline doc comments for {len(df)} PRs × 3 models")
print("=" * 60)

for pr_idx, (_, row) in enumerate(df.iterrows()):
    repo = row.get("repo", "unknown")
    lang = row.get("lang", "unknown")
    print(f"\n[PR {pr_idx + 1}/{len(df)}] {repo}  ({lang})")

    prompt = build_prompt(row)

    for model_name, caller in CALLERS.items():
        t0 = time.time()
        output = caller(prompt)
        output = clean_output(output)
        elapsed = time.time() - t0

        is_error = output.startswith("ERROR") or output.startswith("SKIPPED")
        preview  = output[:100].replace("\n", " ")
        icon     = "✗" if is_error else "✓"
        print(f"  {model_name:<14} {icon} ({elapsed:.1f}s)  {preview}...")

        results.append({
            "pr_id":              pr_idx,
            "repo":               repo,
            "lang":               lang,
            "doc_style":          get_doc_style(lang),
            "model":              model_name,
            "model_id":           {"groq": MODEL_GROQ, "gemini": MODEL_GEMINI,
                                   "huggingface": MODEL_HF}[model_name],
            "prompt":             prompt,
            "generated":          output,
            "ground_truth_patch": get_ground_truth_patch(row),
            "status":             "error" if is_error else "ok",
            # Manual evaluation columns (fill in CSV after running)
            # Scale: 0=no, 1=partial, 2=yes  |  hallucination: 0=none, 2=major
            "correct":       "",
            "useful":        "",
            "hallucination": "",
            "format_correct": "",   # new: is the doc format right for the language?
            "placement":     "",    # new: is placement suggestion sensible?
            "readability":   "",
            "overall_score": "",
            "notes":         "",
        })

        time.sleep(SLEEP_BETWEEN)

    # Crash-safe save after every PR
    OUTPUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

# ============================================================
# SAVE CSV
# ============================================================

print("\n" + "=" * 60)
print("Saving comparison table...")
print("=" * 60)

csv_cols = ["pr_id", "repo", "lang", "doc_style", "model", "model_id",
            "generated", "ground_truth_patch", "status",
            "correct", "useful", "hallucination", "format_correct",
            "placement", "readability", "overall_score", "notes"]
pd.DataFrame(results)[csv_cols].to_csv(OUTPUT_CSV, index=False)

# ============================================================
# SUMMARY
# ============================================================

ok  = sum(1 for r in results if r["status"] == "ok")
err = sum(1 for r in results if r["status"] == "error")

print("\n" + "=" * 60)
print("GENERATION COMPLETE — EXP-001 v2")
print("=" * 60)
print(f"PRs processed    : {len(df)}")
print(f"Total outputs    : {len(results)}")
print(f"Successful       : {ok}")
print(f"Errors/skipped   : {err}")
print(f"JSONL saved      : {OUTPUT_JSONL}")
print(f"CSV saved        : {OUTPUT_CSV}")
print()
print("Next steps:")
print("  1. Open comparison_table.csv")
print("  2. For each row: fill correct/useful/hallucination/format_correct/placement/readability/overall_score")
print("  3. Run evaluate_and_plot.py to generate figures for the presentation")