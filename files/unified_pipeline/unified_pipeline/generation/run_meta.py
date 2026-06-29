"""
generation/run_meta.py — ITERATIVE PROMPT SELF-REFINEMENT (a refinement LAYER).

⚠️ TERMINOLOGY (important for the paper — avoid the "meta-prompting" mislabel):
This module implements ITERATIVE SELF-REFINEMENT of the instruction prompt in the
sense of Self-Refine (Madaan et al., 2023) and White et al. (2023) §F (Question
Refinement): the SAME LLM rewrites its own task prompt for a few rounds under a
fixed output-format constraint. This is NOT the multi-agent meta-prompting system
of Suzgun & Kalai (2024) (conductor + expert roles + task decomposition). We keep
the method tag "+meta" for continuity with earlier logs/slides, but describe it in
the paper as "iterative prompt self-refinement", not "meta-prompting".

Per supervisor feedback (Jun 11 / Jun 25): refinement is applied PER PATTERN with the
SAME LLM used for generation, and the number of rounds is NOT a fixed magic number —
it is applied until the prompt is satisfactory (manual check), typically 2-3 rounds.

To avoid a black box, meta_refine_prompt() returns a STRUCTURED trace: for every round
it records the prompt, its length, and the line-level diff vs the previous round, so
the refinement is fully reproducible and analysable (what was added/removed each round).
"""
from __future__ import annotations
import json, time, sys, difflib, re
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
import config as C
from generation import llm_clients as LLM
from generation.generate import _clean
from generation import prompts as P


# жорстке обмеження формату, яке завжди має лишатися в кінці промпта
_FORMAT_GUARD = ("\n\nOutput ONLY the Javadoc comment block (/** ... */). "
                 "Do not add explanations, prose, or the method code.")


def _approx_tokens(s: str) -> int:
    """Cheap token proxy (whitespace + punctuation) for length-controlled analysis."""
    return len(re.findall(r"\w+|[^\w\s]", s))


def _prompt_diff(prev: str, cur: str) -> dict:
    """Line-level diff between two prompt versions: what the refine step added/removed."""
    added, removed = [], []
    for line in difflib.unified_diff(prev.splitlines(), cur.splitlines(), lineterm=""):
        if line.startswith("+") and not line.startswith("+++"):
            added.append(line[1:].strip())
        elif line.startswith("-") and not line.startswith("---"):
            removed.append(line[1:].strip())
    return {"added": [a for a in added if a], "removed": [r for r in removed if r]}


def meta_refine_prompt(base_prompt, call_fn, rounds=2, converge=0.97, return_trace=False):
    """Iteratively ask the SAME model to improve the prompt, PRESERVING output format
    (re-append a hard format guard each round). Returns a STRUCTURED trace so the
    refinement is not a black box.

    trace = [
      {"round": 0, "prompt": ..., "chars": .., "tokens": .., "diff": None},      # base
      {"round": 1, "prompt": ..., "chars": .., "tokens": .., "diff": {added,removed},
       "ratio": <similarity vs prev>, "format_guard_reattached": bool},
      ...
    ]
    """
    current = base_prompt
    trace = [{"round": 0, "prompt": base_prompt,
              "chars": len(base_prompt), "tokens": _approx_tokens(base_prompt),
              "diff": None}]
    for i in range(1, rounds + 1):
        improved = call_fn(P.question_refinement(current))
        if not LLM.is_ok(improved):
            break
        improved = improved.strip()
        reattached = "Output ONLY" not in improved          # refine dropped the constraint?
        if reattached:
            improved = improved + _FORMAT_GUARD
        ratio = difflib.SequenceMatcher(None, current, improved).ratio()
        trace.append({
            "round": i, "prompt": improved,
            "chars": len(improved), "tokens": _approx_tokens(improved),
            "diff": _prompt_diff(current, improved),
            "ratio": round(ratio, 3),
            "format_guard_reattached": reattached,
        })
        current = improved
        if ratio >= converge:           # converged → stop early (manual-stop proxy)
            break
    return (current, trace) if return_trace else current


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
