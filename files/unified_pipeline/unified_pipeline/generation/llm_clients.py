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

_BACKENDS = {"groq": _groq, "gemini": _gemini, "hf": _hf,
             "hf_text": _hf_text, "deepseek": _deepseek}

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
