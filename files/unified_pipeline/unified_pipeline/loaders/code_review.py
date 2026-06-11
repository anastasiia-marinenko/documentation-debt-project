"""
loaders/code_review.py — Lin et al. (ICSE 2026) Code Review dataset.
REAL loader: reuses your extract_doc_pairs.py logic (added-Javadoc in PR diffs).
Input: train_complete.jsonl  →  Java method + the Javadoc the developer added.
"""
from __future__ import annotations
import json, re
from typing import Iterator
from .base import BaseLoader
from schema import DocPair, TASK_DOC_GENERATION

MIN_JAVADOC_LINES = 3

def _extract_added_javadoc(patch: str) -> list[str]:
    blocks, current, in_block = [], [], False
    for raw in patch.split("\n"):
        if raw.startswith(("+++", "---")):
            continue
        added = raw.startswith("+")
        content = raw[1:] if (added or raw.startswith(" ")) else ""
        s = content.strip()
        if added:
            if s.startswith("/**"):
                in_block, current = True, [content]
            elif in_block and (s.startswith("*") or s == "*/"):
                current.append(content)
                if s.endswith("*/"):
                    if len(current) >= MIN_JAVADOC_LINES:
                        blocks.append("\n".join(current))
                    current, in_block = [], False
        elif in_block:
            in_block, current = False, []
    return blocks

_SIG = re.compile(r"^\s*(public|private|protected|static|final|synchronized|abstract|default).*\(.*\)\s*(\{|throws)")

def _method_from_oldf(oldf: str) -> str:
    if not oldf:
        return ""
    lines = oldf.split("\n")
    start = next((i for i, l in enumerate(lines) if _SIG.match(l)), None)
    if start is None:
        return ""
    body, depth, started = [], 0, False
    for l in lines[start:start + 80]:
        body.append(l); depth += l.count("{") - l.count("}")
        if "{" in l: started = True
        if started and depth <= 0: break
    return "\n".join(body)


class CodeReviewLoader(BaseLoader):
    name = "code_review"

    def _iter_raw(self) -> Iterator[dict]:
        with open(self.raw_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)

    def _to_pair(self, raw: dict) -> DocPair | None:
        # CONFIRM these field names against your jsonl: lang / patch / oldf / repo
        if (raw.get("lang") or raw.get("language", "")).lower() != "java":
            return None
        patch = raw.get("patch") or raw.get("diff", "")
        oldf  = raw.get("oldf")  or raw.get("old_content", "")
        docs  = _extract_added_javadoc(patch)
        if not docs:
            return None
        code = _method_from_oldf(oldf)
        if not code:
            return None
        return DocPair(
            dataset=self.name, task_type=TASK_DOC_GENERATION, language="java",
            code_unit=code, reference_doc=docs[0],
            repo=raw.get("repo", raw.get("project", "unknown")),
        )
