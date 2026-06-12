"""
generation/run_meta.py — EXP-004b: META-PROMPTING AS A REFINEMENT LAYER.

Per supervisor feedback (Jun 11): meta-prompting is NOT a standalone variant.
For EACH prompt pattern (Persona, Template, Recipe, ...) we:
  1) build the pattern's base prompt,
  2) ask the SAME LLM to improve that prompt iteratively (2-3 rounds, stop when
     it converges) — White et al. 2023 §F (Question Refinement) applied per pattern,
  3) generate documentation with the improved prompt.
The improving model and the generating model are the SAME (a meta-prompting rule:
if you refine with model X you must generate with model X).

Output variant = "<pattern>+meta"  (e.g. "v4_persona+meta") so the prompt-study
table shows each pattern WITH and WITHOUT its meta-refinement, side by side.
"""
from __future__ import annotations
import json, time, sys, difflib
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
import config as C
from generation import llm_clients as LLM
from generation.generate import _clean
from generation import prompts as P


def meta_refine_prompt(base_prompt, call_fn, rounds=2, converge=0.97):
    """Iteratively ask the SAME model to improve the prompt. Stop early when a new
    version is ~identical to the previous one (reasonably satisfied)."""
    current = base_prompt
    for _ in range(rounds):
        improved = call_fn(P.question_refinement(current))
        if not LLM.is_ok(improved):
            break
        improved = improved.strip()
        if difflib.SequenceMatcher(None, current, improved).ratio() >= converge:
            current = improved
            break                      # converged -> stop refining
        current = improved
    return current


def run_meta(pairs, model_key="groq", patterns=None, rounds=2, out_path=None):
    out_path = Path(out_path or (C.DATA_RESULTS / "meta_results.jsonl"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    patterns = patterns or list(C.PROMPT_VARIANTS)        # refine every pattern
    spec = C.MODELS[model_key]
    provider, model = spec["provider"], spec["model"]
    call_fn = lambda pr: LLM.call(provider, model, pr)

    rows = []
    for i, p in enumerate(pairs):
        for pattern in patterns:
            base = P.build_prompt(p.code_unit, p.language, pattern)
            improved = meta_refine_prompt(base, call_fn, rounds=rounds)
            out = LLM.call(provider, model, improved)
            rows.append({
                "pair_id": p.pair_id, "dataset": p.dataset, "repo": p.repo,
                "language": p.language, "task_type": p.task_type,
                "model": model_key, "model_id": model,
                "variant": f"{pattern}+meta",
                "reference_doc": p.reference_doc,
                "generated": _clean(out),
                "status": "ok" if LLM.is_ok(out) else "fail",
            })
        with open(out_path, "w", encoding="utf-8") as f:       # crash-safe
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        time.sleep(_sleep(provider))
        print(f"[{i+1}/{len(pairs)}] meta-refined {len(patterns)} patterns on {p.dataset}:{p.repo[:20]}")
    print(f"\nSaved {len(rows)} rows -> {out_path}")
    return rows, out_path


def _sleep(provider):
    return {"groq": C.SLEEP_GROQ, "gemini": C.SLEEP_GEMINI, "hf": C.SLEEP_HF,
            "hf_text": C.SLEEP_HF_HEAVY, "deepseek": C.SLEEP_GROQ}.get(provider, 0.3)
