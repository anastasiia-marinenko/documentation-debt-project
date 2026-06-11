"""
loaders/pentacet.py — PENTACET code-comment dataset.
ADAPTER: large code-comment corpus. CONFIRM columns against release.
"""
from __future__ import annotations
import csv, json
from pathlib import Path
from typing import Iterator
from .base import BaseLoader
from schema import DocPair, TASK_DOC_GENERATION


class PentacetLoader(BaseLoader):
    name = "pentacet"

    def _iter_raw(self) -> Iterator[dict]:
        p = self.raw_path
        files = (list(p.glob("*.csv")) + list(p.glob("*.jsonl"))) if p.is_dir() else [p]
        for fp in files:
            if str(fp).endswith(".jsonl"):
                with open(fp, encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            yield json.loads(line)
            elif str(fp).endswith(".csv"):
                with open(fp, encoding="utf-8", newline="") as f:
                    for row in csv.DictReader(f):
                        yield row

    def _to_pair(self, raw: dict) -> DocPair | None:
        code = raw.get("code") or raw.get("snippet", "")
        doc  = raw.get("comment") or raw.get("text", "")
        if not code or not doc:
            return None
        return DocPair(self.name, TASK_DOC_GENERATION, "java", code, doc,
                       repo=raw.get("repo", "pentacet"))
