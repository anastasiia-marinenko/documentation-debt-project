"""
config.py — single place to set paths, models, datasets, metrics, sampling.
Matches the conventions already used in exp_doc_pairs_llm.py.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
DATA_RAW       = ROOT / "data" / "raw"          # put each dataset's raw files here
DATA_PROCESSED = ROOT / "data" / "processed"    # normalized DocPair parquet/json
DATA_RESULTS   = ROOT / "data" / "results"      # generation outputs + metrics
for p in (DATA_RAW, DATA_PROCESSED, DATA_RESULTS):
    p.mkdir(parents=True, exist_ok=True)

load_dotenv(ROOT / ".env")
GROQ_API_KEY   = os.getenv("GROQ_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
HF_TOKEN       = os.getenv("HF_TOKEN", "")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")  # OpenAI-compatible endpoint
SLEEP_HF_HEAVY   = 1.0  # serverless cold-start for larger open models

# ── Datasets (your scope; README/Gao excluded — different granularity) ──
#   raw_path is where the loader expects to find the dataset files.
DATASETS = {
    # --- the 5 active datasets ---
    "code_review":        {"raw": DATA_RAW / "train_complete.jsonl",       "lang": "java"},  # Lin et al. 2026
    "funcom":             {"raw": DATA_RAW / "funcom",                     "lang": "java"},  # dats.train + coms.train (also reads data/raw root)
    "tesoro":             {"raw": DATA_RAW / "tesoro_comment.json",        "lang": "java"},  # put tesoro_comment.json here
    "codereval":          {"raw": DATA_RAW / "CEJavaRaw.jsonl",            "lang": "java"},  # input / docstring
    "robustness_copilot": {"raw": DATA_RAW / "robustness_copilot.csv",     "lang": "java"},  # body / javaDoc
    # --- excluded from current scope (kept for later) ---
    # "dome":             {"raw": DATA_RAW / "dome",                       "lang": "java"},
    # "pentacet":         {"raw": DATA_RAW / "pentacet",                   "lang": "java"},
}
MAX_PAIRS_PER_DATASET = 25   # keep small for free-tier; raise once tokens available

# ── Models ──────────────────────────────────────────────────────────────
#   provider routing:
#     groq      -> Groq API
#     gemini    -> Google GenAI
#     hf        -> HF Inference, chat_completion  (instruct/chat models)
#     hf_text   -> HF Inference, text_generation  (base completion models)
#     deepseek  -> DeepSeek OpenAI-compatible API (needs DEEPSEEK_API_KEY)
#   Enable a model by flipping "enabled": True. Open-weight models run via the
#   HuggingFace serverless Inference API using the same HF_TOKEN you already use.
MODELS = {
    # already-working baselines
    "groq":        {"provider": "groq",  "model": "llama-3.1-8b-instant",                 "enabled": True},
    "huggingface": {"provider": "hf",    "model": "meta-llama/Meta-Llama-3-8B-Instruct",  "enabled": True},
    "gemini":      {"provider": "gemini","model": "gemini-2.5-flash",                     "enabled": False},  # quota

    # ── open-source / open-weight candidates (from the daily log) ──
    "starchat":    {"provider": "hf",      "model": "HuggingFaceH4/starchat2-15b-v0.1",   "enabled": False},
    "llama2":      {"provider": "hf",      "model": "meta-llama/Llama-2-7b-chat-hf",       "enabled": False},
    "magicoder":   {"provider": "hf",      "model": "ise-uiuc/Magicoder-S-DS-6.7B",        "enabled": False},
    "codellama":   {"provider": "hf",      "model": "codellama/CodeLlama-7b-Instruct-hf",  "enabled": True},   # gated: accept license on HF
    "codegen":     {"provider": "hf_text", "model": "Salesforce/codegen-2B-multi",         "enabled": False},  # base -> text_generation
    "qwen_coder":  {"provider": "hf",      "model": "Qwen/Qwen2.5-Coder-7B-Instruct",      "enabled": True},
    "deepseek":    {"provider": "deepseek","model": "deepseek-coder",                      "enabled": True},   # needs DEEPSEEK_API_KEY
    # PanGu-Coder (Huawei) is NOT openly hosted on HF — needs special access.
    # Add it here with a provider once you obtain an endpoint.
    "pangu_coder": {"provider": "hf",      "model": "REPLACE_WITH_ACCESSIBLE_ENDPOINT",    "enabled": False},
}

MAX_TOKENS   = 512
MAX_TOKENS_VERBOSE = 2048      # gemini-style verbose models
TEMPERATURE  = 0.3
SLEEP_GROQ   = 0.3
SLEEP_GEMINI = 4.0             # 15 RPM free tier
SLEEP_HF     = 0.5

# ── Metrics to compute (all from the daily log) ──
METRICS = ["rouge1", "rougeL", "bleu", "meteor", "bertscore_f1", "cosine"]

# ── Prompt strategies for the prompt-engineering experiment ──
PROMPT_VARIANTS = ["v1_minimal", "v2_structured", "v3_template",
                   "v4_persona", "v5_cognitive"]
DEFAULT_PROMPT = "v3_template"

# ── IRA sampling ──
IRA_TOTAL_SAMPLES = 383   # statistically significant (95% conf, 5% margin); use sample-size calculator
IRA_SEED = 42
