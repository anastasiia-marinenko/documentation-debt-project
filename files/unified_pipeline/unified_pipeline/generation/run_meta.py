"""
generation/run_meta.py — EXP-004b: the two "prompt-improvement" strategies that are
NOT plain single-shot patterns, evaluated against the reference like any variant:

  variant="self_refine"     -> White et al. 2023 §F / Schmidt §3.6 (Question Refinement):
                               ask the LLM to improve the prompt, then run the improved one.
  variant="meta_prompting"  -> Suzgun & Kalai 2024: Meta Model (conductor) + expert author
                               + fresh-eyes reviewer.

Output rows use the SAME schema as generation_results.jsonl, so evaluate/aggregate
treat them as extra prompt variants. One model only (default groq) to stay comparable.
"""
from __future__ import annotations
import json, time, sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
import config as C
from generation import llm_clients as LLM
from generation.generate import _clean
from generation import prompts as P


def run_meta(pairs, model_key="groq", out_path=None):
    out_path = Path(out_path or (C.DATA_RESULTS / "meta_results.jsonl"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    spec = C.MODELS[model_key]
    provider, model = spec["provider"], spec["model"]
    call_fn = lambda pr: LLM.call(provider, model, pr)

    rows = []
    for i, p in enumerate(pairs):
        # 1) self-refinement (Question Refinement): improve a base prompt, then run it
        base = P.build_prompt(p.code_unit, p.language, "v1_minimal")
        improved = LLM.call(provider, model, P.question_refinement(base))
        refined_out = LLM.call(provider, model, improved) if LLM.is_ok(improved) else improved

        # 2) true meta-prompting (Suzgun & Kalai): conductor + expert + fresh-eyes reviewer
        meta_out = P.meta_prompting(p.code_unit, p.language, call_fn)

        for variant, out in (("self_refine", refined_out), ("meta_prompting", meta_out)):
            rows.append({
                "pair_id": p.pair_id, "dataset": p.dataset, "repo": p.repo,
                "language": p.language, "task_type": p.task_type,
                "model": model_key, "model_id": model, "variant": variant,
                "reference_doc": p.reference_doc,
                "generated": _clean(out),
                "status": "ok" if LLM.is_ok(out) else "fail",
            })
        with open(out_path, "w", encoding="utf-8") as f:      # crash-safe
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        time.sleep(_sleep(provider))
        print(f"[{i+1}/{len(pairs)}] meta+self_refine {p.dataset}:{p.repo[:24]}")
    print(f"\nSaved {len(rows)} rows -> {out_path}")
    return rows, out_path


def _sleep(provider):
    return {"groq": C.SLEEP_GROQ, "gemini": C.SLEEP_GEMINI, "hf": C.SLEEP_HF,
            "hf_text": C.SLEEP_HF_HEAVY, "deepseek": C.SLEEP_GROQ}.get(provider, 0.3)
