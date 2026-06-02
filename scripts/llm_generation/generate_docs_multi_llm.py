"""
scripts/llm_generation/generate_docs_multi_llm.py

EXP-001 Configuration:
  Models       : llama-3.1-8b-instant (Groq), gemini-2.5-flash (Gemini),
                 meta-llama/Meta-Llama-3-8B-Instruct (HuggingFace)
  max_tokens   : 512
  temperature  : 0.3 (Groq/Gemini), 0.2 (HF)
  Prompt ver   : v2
  Input        : code diff + commit messages + README context
  Output format: DOCUMENTATION UPDATE / REASON
  Date         : 2026-05-29
"""

import json
import os
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv   # ← використовуємо тільки імпортовану, не перевизначаємо

load_dotenv()   # завантажує .env один раз тут

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

MAX_TOKENS    = 1024
TEMPERATURE   = 0.3
SLEEP_BETWEEN = 4   # секунди між викликами (ліміт Gemini 15 RPM)

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
# PROMPT BUILDER  v2
# ============================================================

def build_prompt(row: pd.Series) -> str:
    import json

    repo      = row.get("repo", "unknown")
    lang      = row.get("lang", "unknown")
    msgs      = list(row.get("messages", []))[:5]

    # ✅ Правильно читаємо readme_patches_json
    readme_patch_text = ""
    old_readme_text = ""
    try:
        patches = json.loads(row.get("readme_patches_json", "[]"))
        for p in patches[:1]:
            readme_patch_text = p.get("patch", "")[:1500]
            old_readme_text = p.get("old_content", "")[:2000]
    except Exception:
        pass

    # Code patches
    code_text = ""
    try:
        code_patches = json.loads(row.get("code_patches_json", "[]"))
        code_text = "\n\n".join(str(p)[:600] for p in code_patches[:2])
    except Exception:
        pass

    msg_text = "\n".join(f"- {m}" for m in msgs) if msgs else "(none)"

    return f"""You are a software documentation expert helping maintain README files.

    Repository  : {repo}
    Language    : {lang}

    Commit messages from this pull request:
    {msg_text}

    README file BEFORE this pull request:
    ---
    {old_readme_text if old_readme_text else "(not available)"}
    ---

    Code changes (diff):
    ---
    {code_text if code_text else "(none)"}
    ---

    Actual README change made in this PR (ground truth):
    ---
    {readme_patch_text if readme_patch_text else "(not available)"}
    ---

    Task:
    1. Write a concrete README update (2-4 sentences) that documents what changed.
    2. In one sentence, explain WHY this update is needed.

    Respond ONLY in this format:
    DOCUMENTATION UPDATE:
    <the text to add or change in the README>

    REASON:
    <why this update is needed>"""

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
                max_output_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
            ),
        )
        return response.text.strip()
    except ImportError:
        return "ERROR: run: pip install google-genai"
    except Exception as e:
        err = str(e)
        if "429" in err:
            return "ERROR (Gemini): 429 quota exceeded — зачекай до завтра або заміни модель на gemini-1.5-flash-8b"
        if "404" in err:
            return f"ERROR (Gemini): модель не знайдена — перевір MODEL_GEMINI. {e}"
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
        # Прибираємо шаблонний префікс Llama
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
            return ("ERROR (HF): 403 — токен не має дозволу на Inference API. "
                    "huggingface.co/settings/tokens → Fine-grained → увімкни Inference API")
        if "400" in err:
            return "ERROR (HF): 400 — модель не підтримує цей тип запиту"
        return f"ERROR (HuggingFace): {e}"

# ============================================================
# LOAD
# ============================================================

print("\n" + "=" * 60)
print("Loading golden set...")
print("=" * 60)

def clean_output(text: str) -> str:
    """Обрізає все що йде до DOCUMENTATION UPDATE:"""
    marker = "DOCUMENTATION UPDATE:"
    idx = text.find(marker)
    if idx != -1:
        return text[idx:].strip()
    return text.strip()

if not INPUT_FILE.exists():
    print(f"ERROR: {INPUT_FILE} not found.")
    print("Run scripts/preprocessing/select_golden_set.py first.")
    exit(1)

df = pd.read_parquet(INPUT_FILE)
print(f"Loaded {len(df)} PRs for generation")

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
print("Running generation for all PRs × 3 models")
print("=" * 60)

for pr_idx, (_, row) in enumerate(df.iterrows()):
    repo = row.get("repo", "unknown")
    print(f"\n[PR {pr_idx + 1}/{len(df)}] {repo}")

    prompt = build_prompt(row)

    import json


    def get_readme_patch(row):
        try:
            patches = json.loads(row.get("readme_patches_json", "[]"))
            return patches[0].get("patch", "") if patches else ""
        except Exception:
            return ""

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
            "lang":               row.get("lang", ""),
            "model":              model_name,
            "model_id":           {"groq": MODEL_GROQ, "gemini": MODEL_GEMINI,
                                   "huggingface": MODEL_HF}[model_name],
            "prompt":             prompt,
            "generated":          output,
            "ground_truth_patch":  get_readme_patch(row),
            "status":             "error" if is_error else "ok",
            # Manual evaluation — fill these in CSV
            "correct":       "",
            "useful":        "",
            "hallucination": "",
            "missing_info":  "",
            "relevance":     "",
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

csv_cols = ["pr_id", "repo", "lang", "model", "model_id",
            "generated", "ground_truth_patch", "status",
            "correct", "useful", "hallucination", "missing_info",
            "relevance", "readability", "overall_score", "notes"]
pd.DataFrame(results)[csv_cols].to_csv(OUTPUT_CSV, index=False)

# ============================================================
# SUMMARY
# ============================================================

ok  = sum(1 for r in results if r["status"] == "ok")
err = sum(1 for r in results if r["status"] == "error")

print("\n" + "=" * 60)
print("GENERATION COMPLETE")
print("=" * 60)
print(f"PRs processed    : {len(df)}")
print(f"Total outputs    : {len(results)}")
print(f"Successful       : {ok}")
print(f"Errors/skipped   : {err}")
print(f"JSONL saved      : {OUTPUT_JSONL}")
print(f"CSV saved        : {OUTPUT_CSV}")
print("\nNext: відкрий comparison_table.csv і заповни колонки оцінки вручну")
print("(correct / useful / hallucination / missing_info / relevance / readability / overall_score)")