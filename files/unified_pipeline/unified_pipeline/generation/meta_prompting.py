"""
generation/meta_prompting.py — TRUE meta-prompting via role orchestration.

Distinct from generation/run_meta.py (which does iterative PROMPT refinement and,
per our traces, fails because code-LLMs execute the instruction instead of editing
it). Following Suzgun & Kalai (2024), meta-prompting here does NOT rewrite the
prompt. Instead a SINGLE LLM is invoked several times in different ROLES to improve
the SOLUTION PROCESS:

    conductor  (implicit: the fixed protocol below)
        -> writer   : produce an initial Javadoc for the method
        -> reviewer : critique that Javadoc against a rubric (LLM-as-a-judge)
        -> writer   : revise the Javadoc using the critique
        -> final Javadoc

Key properties (defensible in the paper):
  * SAME model for every role (no extra model needed) — a self-orchestration /
    LLM-as-a-judge setup; we call it "meta-prompting (writer-reviewer)".
  * We improve the answer, not the prompt template, so there is no prompt
    contamination / pattern collapse.
  * The number of review rounds is a parameter (default 1); each round = one
    critique + one revision. We log the full trace for transparency.

If even this does not beat the baseline, that is a clean result: "single-model
writer-reviewer meta-prompting does not improve already well-structured prompts
for Javadoc on code-LLMs" — reported alongside the refinement negative result.
"""
from __future__ import annotations
import re, sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from generation import llm_clients as LLM
from generation.generate import _clean


def _doc_only(text: str) -> str:
    """Keep just the /** ... */ block if present."""
    t = re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL)
    s, e = t.find("/**"), t.rfind("*/")
    return t[s:e + 2].strip() if s != -1 and e != -1 and e > s else t.strip()


_REVIEWER = (
    "You are a senior Java API reviewer. Critique the Javadoc below for the given "
    "method. Check: (1) summary accuracy vs the code, (2) every @param present and "
    "correct, (3) @return correct if the method returns, (4) @throws for thrown "
    "exceptions, (5) no hallucinated behaviour, (6) concise standard Javadoc style. "
    "List concrete problems as short bullet points. If it is already correct and "
    "complete, reply exactly: OK.\n\nMETHOD:\n```java\n{code}\n```\n\nJAVADOC:\n{draft}"
)

_REVISER = (
    "Revise the Javadoc for the method below using the reviewer's notes. Output ONLY "
    "the corrected Javadoc block (/** ... */) — no prose, no code.\n\n"
    "METHOD:\n```java\n{code}\n```\n\nCURRENT JAVADOC:\n{draft}\n\n"
    "REVIEWER NOTES:\n{critique}"
)


def meta_prompting_generate(code, base_prompt_fn, provider, model,
                            max_tokens, rounds=1, return_trace=False):
    """
    Generate a Javadoc via writer -> (reviewer -> writer) x rounds, all with the
    SAME model. `base_prompt_fn(code)` is the pattern's normal prompt (the Stage-1
    winner) used for the FIRST draft, so meta-prompting is compared fairly against
    that same pattern's baseline.
    """
    trace = []

    # 1) WRITER — initial draft using the pattern's own (winning) prompt
    draft = _doc_only(LLM.call(provider, model, base_prompt_fn(code), max_tokens=max_tokens))
    trace.append({"role": "writer", "round": 0, "output": draft})

    for r in range(1, rounds + 1):
        # 2) REVIEWER — critique the current draft (LLM-as-a-judge)
        critique = LLM.call(provider, model,
                            _REVIEWER.format(code=code[:4000], draft=draft),
                            max_tokens=max_tokens)
        critique = re.sub(r"<think>.*?</think>", "", critique or "", flags=re.DOTALL).strip()
        trace.append({"role": "reviewer", "round": r, "output": critique})

        # early stop: reviewer is satisfied
        if critique.strip().upper().startswith("OK") or len(critique.strip()) < 3:
            trace.append({"role": "stop", "round": r, "output": "reviewer satisfied"})
            break

        # 3) WRITER — revise using the critique
        revised = _doc_only(LLM.call(provider, model,
                                     _REVISER.format(code=code[:4000], draft=draft,
                                                     critique=critique),
                                     max_tokens=max_tokens))
        trace.append({"role": "writer", "round": r, "output": revised})
        if revised:
            draft = revised

    return (draft, trace) if return_trace else draft
