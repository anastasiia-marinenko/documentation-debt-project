"""
loaders/dome.py — DOME (Hu et al., 2022) method documentation generation.
ADAPTER: DOME provides (method, comment) pairs. CONFIRM keys against release.
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Iterator
from .base import BaseLoader
from schema import DocPair, TASK_DOC_GENERATION


class DomeLoader(BaseLoader):
    name = "dome"

    def _iter_raw(self) -> Iterator[dict]:
        p = self.raw_path
        files = list(p.glob("*.jsonl")) if p.is_dir() else [p]
        for fp in files:
            with open(fp, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        yield json.loads(line)

    def _to_pair(self, raw: dict) -> DocPair | None:
        code = raw.get("code") or raw.get("method") or raw.get("source", "")
        doc  = raw.get("comment") or raw.get("nl") or raw.get("doc", "")
        if not code or not doc:
            return None
        return DocPair(self.name, TASK_DOC_GENERATION, "java", code, doc,
                       repo=raw.get("repo", "dome"))
