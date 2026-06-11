"""
generation/prompts.py — prompt variants for EXP-004, each a faithful instantiation
of a NAMED pattern from the prompt-engineering papers in the folder.

Grounding (paper -> section -> variant):
  White et al. (2023), "A Prompt Pattern Catalog to Enhance Prompt Engineering with ChatGPT"
      §E Persona            -> v4_persona
      §F Question Refinement-> question_refinement()      (a.k.a. "Prompt Improvement")
      §H Cognitive Verifier -> v5_cognitive
      §J Template           -> v3_template
      §Q Recipe             -> v2_structured
  Schmidt et al. (2023), "Towards a Catalog of Prompt Patterns ..." (Ada Letters)
      methodological frame: prompt patterns as a *discipline* with quality attributes;
      patterns-as-abstractions. Justifies systematically varying named patterns.
  Suzgun & Kalai (2024), "Meta-Prompting: ... Task-Agnostic Scaffolding"
      meta_prompting()  -> Meta Model (conductor) + expert instances + "fresh eyes".
      v1_minimal is the paper's "standard prompting" baseline.

NOTE: v1/v2/v4 keep their original wording; v3 and v5 were made faithful to the
named patterns (literal template skeleton; sub-question verifier), so re-run EXP-004
to get numbers that match these definitions.
"""

def _doc_style(language: str) -> str:
    return {"java": "Javadoc (/** ... */)",
            "python": "a docstring (\"\"\" ... \"\"\")",
            "c#": "XML doc comments (/// <summary>...)"}.get(language.lower(), "a doc comment")


# --- v1: baseline (no pattern) = "standard prompting" (Suzgun & Kalai 2024) ----------
def v1_minimal(code, language):
    return f"Write {_doc_style(language)} for this {language} code:\n\n{code[:1500]}"


# --- v2: RECIPE pattern (White et al. 2023, §Q) --------------------------------------
# "I would like to achieve X; I know I need steps A,B,C; provide the complete result;
#  fill in any missing steps; identify/omit any unnecessary steps."
def v2_structured(code, language):
    return (f"I want complete {_doc_style(language)} for the {language} method below.\n"
            f"I know it should include: a one-sentence summary; @param for each parameter; "
            f"@return if it returns a value; @throws if it can throw.\n"
            f"Provide the complete doc block, fill in anything missing, and omit any tag "
            f"that does not apply. Output ONLY the doc block.\n\n```{language}\n{code[:1500]}\n```")


# --- v3: TEMPLATE pattern (White et al. 2023, §J) ------------------------------------
# Provide a literal template with placeholders; preserve it; omit placeholders that
# do not apply. (Faithful Template = a skeleton, not just a list of requirements.)
def v3_template(code, language):
    return (f"I am going to provide a template for your output. Preserve this exact "
            f"{_doc_style(language)} structure and replace each ALL-CAPS placeholder with "
            f"content inferred from the method. Omit any placeholder line that does not "
            f"apply. Output ONLY the filled template.\n\n"
            f"/**\n"
            f" * SUMMARY_ONE_SENTENCE\n"
            f" * @param PARAM_NAME PARAM_DESCRIPTION\n"
            f" * @return RETURN_DESCRIPTION\n"
            f" * @throws EXCEPTION_TYPE CONDITION\n"
            f" */\n\n"
            f"```{language}\n{code[:1500]}\n```")


# --- v4: PERSONA pattern (White et al. 2023, §E) -------------------------------------
# "Act as persona X; provide outputs that persona X would create."
def v4_persona(code, language):
    return (f"Act as a senior {language} engineer and meticulous API-doc author known "
            f"for concise, accurate documentation. Provide the {_doc_style(language)} such "
            f"an author would write for the method below, documenting only what the code "
            f"actually does.\nOutput ONLY the doc block.\n\n```{language}\n{code[:1500]}\n```")


# --- v5: COGNITIVE VERIFIER pattern (White et al. 2023, §H) --------------------------
# Subdivide into sub-questions, answer them, then combine into the final answer.
def v5_cognitive(code, language):
    return (f"To document the {language} method below, first answer these sub-questions, "
            f"then combine the answers into the final documentation:\n"
            f"(1) What single responsibility does this method have?\n"
            f"(2) What is each parameter and what does it represent?\n"
            f"(3) What does it return, and under what conditions does it throw?\n"
            f"After answering internally, output ONLY the final {_doc_style(language)} block "
            f"(do not show the sub-answers).\n\n```{language}\n{code[:1500]}\n```")


VARIANTS = {
    "v1_minimal": v1_minimal, "v2_structured": v2_structured, "v3_template": v3_template,
    "v4_persona": v4_persona, "v5_cognitive": v5_cognitive,
}

def build_prompt(code, language, variant="v3_template"):
    return VARIANTS[variant](code, language)


# --- QUESTION REFINEMENT / Prompt Improvement (White §F; Schmidt §3.6) ---------------
# This is what was previously (mis)labelled "meta": ask the LLM to improve the prompt.
def question_refinement(base_prompt):
    """White et al. 2023 §F: suggest a better version of the prompt, then use it."""
    return ("Within the scope of code documentation, suggest a better version of the "
            "prompt below so a code LLM produces more accurate, concise documentation. "
            "Return ONLY the improved prompt text.\n\n---\n" + base_prompt + "\n---")

# Back-compat alias (old name) so existing scripts don't break:
meta_improve_prompt = question_refinement


# --- TRUE META-PROMPTING (Suzgun & Kalai 2024) --------------------------------------
# Meta Model (conductor) drafts -> Expert API-doc author writes -> Expert reviewer with
# "fresh eyes" verifies/corrects -> conductor finalises. Each expert is the same LM
# prompted only on the conductor's instruction (fresh eyes = no full history).
def meta_prompting(code, language, call_fn):
    """
    Suzgun & Kalai (2024) meta-prompting, minimal faithful form.
      call_fn(prompt:str) -> str   (a single LM call, e.g. llm_clients.call(...))
    Returns the final documentation string.
    """
    snippet = code[:1500]
    style = _doc_style(language)

    # 1) Conductor decomposes & instructs the first expert.
    writer_instruction = (
        f"You are an expert {language} API-documentation author. Write {style} for the "
        f"method below, documenting only what the code does. Output ONLY the doc block.\n\n"
        f"```{language}\n{snippet}\n```")
    draft = call_fn(writer_instruction)

    # 2) Fresh-eyes expert verifies (prompted ONLY on the draft + code, no history).
    reviewer_instruction = (
        f"You are an expert {language} documentation reviewer with fresh eyes. Check the "
        f"draft doc against the code for accuracy, hallucinated params/returns, and missing "
        f"@throws. Return ONLY a corrected final {style} block.\n\n"
        f"CODE:\n```{language}\n{snippet}\n```\n\nDRAFT:\n{draft}")
    final = call_fn(reviewer_instruction)
    return final.strip()
