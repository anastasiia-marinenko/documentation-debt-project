"""
loaders/codereval.py — CoderEval (Yu et al. 2024), function-level benchmark.
Real schema (from exploration): CEJavaRaw.jsonl (JSONL), 239 Java functions.
    input      -> function source code
    docstring  -> reference documentation
    signature  -> method signature (kept in meta)
    question_id-> id
"""
from __future__ import annotations
import json
from typing import Iterator
from .base import BaseLoader
from schema import DocPair, TASK_DOC_GENERATION


class CoderEvalLoader(BaseLoader):
    name = "codereval"

    def _iter_raw(self) -> Iterator[dict]:
        text = open(self.raw_path, encoding="utf-8").read().strip()
        # try whole-file JSON (array or {"RECORDS": [...]})
        try:
            data = json.loads(text)
            records = data.get("RECORDS", data) if isinstance(data, dict) else data
            for r in records:
                yield r
            return
        except json.JSONDecodeError:
            pass
        # fall back to JSONL (CEJavaRaw.jsonl = one object per line)
        for line in text.splitlines():
            line = line.strip()
            if line:
                yield json.loads(line)

    def _to_pair(self, raw: dict) -> DocPair | None:
        lang = (raw.get("language") or raw.get("lang", "")).lower()
        if lang and "java" not in lang:
            return None
        code = raw.get("input") or raw.get("code") or raw.get("function") or raw.get("solution", "")
        doc  = raw.get("docstring") or raw.get("comment") or raw.get("human_label", "")
        if not code or not doc:
            return None
        pair = DocPair(self.name, TASK_DOC_GENERATION, "java", code, doc,
                       repo=str(raw.get("question_id", raw.get("project", "codereval"))))
        if raw.get("signature"):
            pair.meta["signature"] = raw["signature"]
        return pair
