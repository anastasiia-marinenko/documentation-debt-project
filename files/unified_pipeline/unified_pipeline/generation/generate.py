"""
generation/generate.py — run generation over normalized DocPairs.
Produces one row per (pair x model x prompt_variant). Output: results jsonl.
"""
from __future__ import annotations
import json, time, sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
import config as C
from generation import llm_clients as LLM
from generation.prompts import build_prompt

_SLEEP = {"groq": C.SLEEP_GROQ, "gemini": C.SLEEP_GEMINI, "hf": C.SLEEP_HF,
          "hf_text": C.SLEEP_HF_HEAVY, "deepseek": C.SLEEP_GROQ}


def _clean(text: str) -> str:
    """Strip markdown fences so metrics compare doc-to-doc."""
    t = str(text).strip()
    if t.startswith("```"):
        t = t.split("```", 2)[1] if t.count("```") >= 2 else t.strip("`")
        t = t.split("\n", 1)[-1] if t[:6].lower() in ("java\n", "python") else t
    return t.strip()


def run(pairs, models=None, variants=None, out_path=None):
    models = models or [m for m, v in C.MODELS.items() if v["enabled"]]
    variants = variants or [C.DEFAULT_PROMPT]
    out_path = Path(out_path or (C.DATA_RESULTS / "generation_results.jsonl"))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for i, p in enumerate(pairs):
        for variant in variants:
            prompt = build_prompt(p.code_unit, p.language, variant)
            for mkey in models:
                spec = C.MODELS[mkey]
                out = LLM.call(spec["provider"], spec["model"], prompt)
                rows.append({
                    "pair_id": p.pair_id, "dataset": p.dataset, "repo": p.repo,
                    "language": p.language, "task_type": p.task_type,
                    "model": mkey, "model_id": spec["model"], "variant": variant,
                    "reference_doc": p.reference_doc,
                    "generated": _clean(out),
                    "status": "ok" if LLM.is_ok(out) else "fail",
                })
                time.sleep(_SLEEP.get(spec["provider"], 0.3))
        # crash-safe incremental save
        with open(out_path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[{i+1}/{len(pairs)}] {p.dataset}:{p.repo[:24]}")
    print(f"\nSaved {len(rows)} rows -> {out_path}")
    return rows, out_path
