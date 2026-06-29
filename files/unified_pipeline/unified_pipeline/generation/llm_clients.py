"""
generation/llm_clients.py — one call(provider, model, prompt) for every backend.
Providers: groq, gemini, hf (chat), hf_text (base completion), deepseek.
Open-weight models (StarChat, Llama 2, Magicoder, CodeLlama, CodeGen, Qwen-Coder)
run through the HuggingFace serverless Inference API using your HF_TOKEN.
Each backend degrades gracefully: missing key/lib -> "SKIPPED", error -> "ERROR:...".
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
import config as C
import os, re


def _groq(model, prompt, max_tokens):
    if not C.GROQ_API_KEY: return "SKIPPED"
    from groq import Groq
    r = Groq(api_key=C.GROQ_API_KEY).chat.completions.create(
        model=model, messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens, temperature=C.TEMPERATURE)
    return r.choices[0].message.content.strip()

def _gemini(model, prompt, max_tokens):
    if not C.GEMINI_API_KEY: return "SKIPPED"
    from google import genai
    from google.genai import types
    r = genai.Client(api_key=C.GEMINI_API_KEY).models.generate_content(
        model=model, contents=prompt,
        config=types.GenerateContentConfig(max_output_tokens=max_tokens,
                                            temperature=C.TEMPERATURE))
    return r.text.strip()

def _hf(model, prompt, max_tokens):
    if not C.HF_TOKEN: return "SKIPPED"
    from huggingface_hub import InferenceClient
    r = InferenceClient(api_key=C.HF_TOKEN).chat_completion(
        model=model, messages=[{"role": "user", "content": prompt}], max_tokens=max_tokens)
    return r.choices[0].message.content.strip()

def _hf_text(model, prompt, max_tokens):
    """For BASE completion models (CodeGen, base CodeLlama) that aren't chat."""
    if not C.HF_TOKEN: return "SKIPPED"
    from huggingface_hub import InferenceClient
    out = InferenceClient(api_key=C.HF_TOKEN).text_generation(
        prompt, model=model, max_new_tokens=max_tokens,
        temperature=max(C.TEMPERATURE, 0.01), return_full_text=False)
    return out.strip()

def _deepseek(model, prompt, max_tokens):
    """DeepSeek via its OpenAI-compatible endpoint (needs DEEPSEEK_API_KEY)."""
    if not C.DEEPSEEK_API_KEY: return "SKIPPED"
    from openai import OpenAI
    client = OpenAI(api_key=C.DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")
    r = client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens, temperature=C.TEMPERATURE)
    return r.choices[0].message.content.strip()

def _ollama(model, prompt, max_tokens):
    """Local Ollama via the NATIVE /api/generate endpoint.

    Key fix: thinking models (DeepSeek-R1) are called with think=False, so they
    never spend the token budget on a <think> trace and never truncate before the
    Javadoc block. This stabilises the valid-generation count n AND speeds things up.
    We also keep the model warm in VRAM (keep_alive) and bound num_predict / num_ctx
    so large models run on small GPUs without reloading each call.

    NOTE: models are used at their default (un-quantised-by-us) Ollama weights; we do
    not apply any extra quantisation here — we evaluate the models as shipped.
    """
    import requests, json
    base = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/").removesuffix("/v1")
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "think": False,                       # <-- disable reasoning trace (R1)
        "keep_alive": C.OLLAMA_KEEP_ALIVE,     # keep weights warm between calls
        "options": {
            "temperature": C.TEMPERATURE,
            "num_predict": max_tokens,         # Javadoc is short; 512 is plenty
            "num_ctx": C.OLLAMA_NUM_CTX,       # size context to the prompt, not huge
        },
    }
    last = ""
    for attempt in range(C.MAX_RETRIES):
        try:
            r = requests.post(base + "/api/generate", json=payload,
                              timeout=C.OLLAMA_TIMEOUT)
            r.raise_for_status()
            text = (r.json().get("response") or "").strip()
            # Safety net: strip any stray reasoning even though think=False
            text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
            text = re.sub(r"^.*?</think>", "", text, flags=re.DOTALL).strip()
            if text:
                return text
            last = "ERROR:empty-response"
        except Exception as e:
            last = f"ERROR:{e}"
    return last  # honest failure after MAX_RETRIES (caller logs it, does not crash)

_BACKENDS = {"groq": _groq, "gemini": _gemini, "hf": _hf,
             "hf_text": _hf_text, "deepseek": _deepseek, "ollama": _ollama}

def call(provider, model, prompt, max_tokens=None):
    max_tokens = max_tokens or (C.MAX_TOKENS_VERBOSE if provider == "gemini" else C.MAX_TOKENS)
    fn = _BACKENDS.get(provider)
    if fn is None:
        return f"ERROR:unknown provider {provider}"
    try:
        return fn(model, prompt, max_tokens)
    except Exception as e:
        return f"ERROR:{e}"

def is_ok(text) -> bool:
    return not str(text).startswith(("ERROR", "SKIPPED")) and bool(str(text).strip())
