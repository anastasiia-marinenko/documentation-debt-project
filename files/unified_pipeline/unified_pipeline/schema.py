"""
schema.py — the common representation every dataset is normalized to.

This is the heart of the "general methodology": no matter what task a
dataset was built for (summary, SATD, review, doc mining), each usable
example is reduced to ONE record: a code unit + its reference documentation.
Everything downstream (generation, metrics, IRA) only ever sees DocPair.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict, field
from typing import Optional
import hashlib


# Common task vocabulary across datasets (kept for the comparison table).
TASK_DOC_GENERATION = "doc_generation"   # produce documentation for a code unit
TASK_SUMMARIZATION  = "code_summarization"
TASK_SATD           = "satd_detection"
TASK_REVIEW         = "review_comment"
TASK_INCONSISTENCY  = "code_comment_inconsistency"


@dataclass
class DocPair:
    """One normalized (code, reference-doc) example."""
    dataset: str                 # e.g. "code_review", "funcom"
    task_type: str               # one of TASK_* above
    language: str                # "java", "python", ...
    code_unit: str               # method/function source WITHOUT its doc
    reference_doc: str           # ground-truth documentation (Javadoc/summary/...)
    repo: str = "unknown"
    pair_id: str = ""            # stable id (filled in __post_init__ if empty)
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.pair_id:
            h = hashlib.md5(
                (self.dataset + self.code_unit + self.reference_doc).encode("utf-8")
            ).hexdigest()[:10]
            self.pair_id = f"{self.dataset}:{h}"

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("meta", None)
        return d


def is_valid_pair(code: Optional[str], doc: Optional[str],
                  min_code_lines: int = 3, min_doc_chars: int = 15,
                  min_code_chars: int = 40) -> bool:
    """Shared validity gate so every loader filters consistently.
    Accepts either a multi-line method OR a single long tokenized line
    (Funcom-style), so summarization datasets aren't wrongly dropped."""
    if not code or not doc:
        return False
    code_s, doc_s = code.strip(), doc.strip()
    if len(doc_s) < min_doc_chars:
        return False
    if len(code_s.splitlines()) < min_code_lines and len(code_s) < min_code_chars:
        return False
    return True
