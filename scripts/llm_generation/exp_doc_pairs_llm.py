"""
scripts/llm_generation/exp_doc_pairs_llm.py

Experiment: LLM API models for Java Javadoc generation
Dataset:    Lin et al. (2026) — extracted documentation pairs
            (doc_pairs_sample.json from extract_doc_pairs.py)
Models:     Groq (Llama 3.1) · Gemini 2.5 Flash · HuggingFace (Llama 3)
Metrics:    ROUGE-1 · ROUGE-L · BERTScore F1

Run after extract_doc_pairs.py:
    pip install rouge-score bert-score
    python exp_doc_pairs_llm.py
"""

import json, os, time
from pathlib import Path
from dotenv import load_dotenv

ROOT        = Path(__file__).resolve().parents[2]
INPUT_JSON  = ROOT / "data/processed/doc_pairs_sample.json"
OUT_JSONL   = ROOT / "data/results/exp_doc_pairs_results.jsonl"
OUT_CSV     = ROOT / "data/results/exp_doc_pairs_metrics.csv"

MAX_TOKENS  = 512
TEMPERATURE = 0.3
MODEL_GROQ  = "llama-3.1-8b-instant"
MODEL_GEMINI= "gemini-2.5-flash"
MODEL_HF    = "meta-llama/Meta-Llama-3-8B-Instruct"

load_dotenv(ROOT / ".env")
GROQ_KEY   = os.getenv("GROQ_API_KEY","")
GEMINI_KEY = os.getenv("GEMINI_API_KEY","")
HF_TOKEN   = os.getenv("HF_TOKEN","")

print("API KEYS:", "Groq" if GROQ_KEY else "-",
      "| Gemini" if GEMINI_KEY else "| -",
      "| HF" if HF_TOKEN else "| -")

# ── load pairs ────────────────────────────────────────────────────────
with open(INPUT_JSON, encoding="utf-8") as f:
    pairs = json.load(f)
print(f"Loaded {len(pairs)} pairs\n")

# ── prompt ────────────────────────────────────────────────────────────
def build_prompt(code: str) -> str:
    return f"""You are a Java documentation expert.

Write a complete Javadoc comment for the following Java method.
Include:
- A one-sentence summary describing what the method does
- @param tag for each parameter
- @return tag if the method returns a value
- @throws tag if the method may throw exceptions

Output ONLY the Javadoc block (/** ... */). No explanation, no code.

```java
{code[:1500]}
```"""

# ── LLM callers ───────────────────────────────────────────────────────
def call_groq(p):
    if not GROQ_KEY: return "SKIPPED"
    try:
        from groq import Groq
        r = Groq(api_key=GROQ_KEY).chat.completions.create(
            model=MODEL_GROQ,
            messages=[{"role":"user","content":p}],
            max_tokens=MAX_TOKENS, temperature=TEMPERATURE)
        return r.choices[0].message.content.strip()
    except Exception as e: return f"ERROR:{e}"

def call_gemini(p):
    if not GEMINI_KEY: return "SKIPPED"
    try:
        from google import genai
        from google.genai import types
        r = genai.Client(api_key=GEMINI_KEY).models.generate_content(
            model=MODEL_GEMINI, contents=p,
            config=types.GenerateContentConfig(
                max_output_tokens=MAX_TOKENS, temperature=TEMPERATURE))
        return r.text.strip()
    except Exception as e: return f"ERROR:{e}"

def call_hf(p):
    if not HF_TOKEN: return "SKIPPED"
    try:
        from huggingface_hub import InferenceClient
        r = InferenceClient(api_key=HF_TOKEN).chat_completion(
            model=MODEL_HF,
            messages=[{"role":"user","content":p}],
            max_tokens=MAX_TOKENS)
        return r.choices[0].message.content.strip()
    except Exception as e: return f"ERROR:{e}"

# ── generation loop ───────────────────────────────────────────────────
results = []
OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)

for i, pair in enumerate(pairs):
    prompt = build_prompt(pair["method_code"])
    print(f"[{i+1:02d}/{len(pairs)}] {pair.get('repo','?')[:40]}")

    g = call_groq(prompt);    time.sleep(0.3)
    m = call_gemini(prompt);  time.sleep(4.0)
    h = call_hf(prompt);      time.sleep(0.5)

    for name, out in [("groq",g),("gemini",m),("huggingface",h)]:
        ok = not (str(out).startswith("ERROR") or str(out).startswith("SKIPPED"))
        print(f"   {name:12s} {'✓' if ok else '✗'}")

    results.append({**pair, "prompt":prompt,
                    "groq_output":g, "gemini_output":m,
                    "huggingface_output":h})

    with open(OUT_JSONL,"w",encoding="utf-8") as f:
        for r in results: f.write(json.dumps(r,ensure_ascii=False)+"\n")

# ── metrics ───────────────────────────────────────────────────────────
import pandas as pd
from rouge_score import rouge_scorer
from bert_score import score as bscore

sc = rouge_scorer.RougeScorer(["rouge1","rougeL"], use_stemmer=True)
rows = []
for r in results:
    gt = r["javadoc_gt"]
    for model, out in [("groq",r["groq_output"]),
                       ("gemini",r["gemini_output"]),
                       ("huggingface",r["huggingface_output"])]:
        if str(out).startswith(("ERROR","SKIPPED")) or not str(out).strip():
            continue
        rs = sc.score(gt, out)
        rows.append({"repo":r["repo"],"model":model,
                     "generated":out,"ground_truth":gt,
                     "rouge1":round(rs["rouge1"].fmeasure,4),
                     "rougeL":round(rs["rougeL"].fmeasure,4)})

df = pd.DataFrame(rows)
if len(df):
    _,_,F1 = bscore(df["generated"].tolist(), df["ground_truth"].tolist(),
                    lang="en", verbose=False)
    df["bertscore_f1"] = [round(x.item(),4) for x in F1]
    df.to_csv(OUT_CSV, index=False)

    print("\n" + "="*55)
    print("RESULTS — Lin et al. (2026) Java documentation pairs")
    print("="*55)
    s = df.groupby("model")[["rouge1","rougeL","bertscore_f1"]].mean().round(4)
    s.columns = ["ROUGE-1","ROUGE-L","BERTScore F1"]
    print(s.to_string())
    print("="*55)
    print(f"\nCSV: {OUT_CSV}")
