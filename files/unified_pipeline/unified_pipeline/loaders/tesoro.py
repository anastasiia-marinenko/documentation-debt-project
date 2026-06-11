"""
loaders/tesoro.py — TESORO (SATD detection). Real schema (from exploration):
  file: tesoro_comment.json  (JSONL despite the .json extension)
  cols: id, comment_id, comment, code, classification, isFinished, code_context_*
  classification labels: NONSATD / DESIGN / IMPLEMENTATION / DEFECT / DOCUMENTATION / TEST
We keep the (code, comment) pair; the SATD label is stored in meta.
"""
from __future__ import annotations
import csv, json
from pathlib import Path
from typing import Iterator
from .base import BaseLoader
from schema import DocPair, TASK_SATD


class TesoroLoader(BaseLoader):
    name = "tesoro"

    def _iter_raw(self) -> Iterator[dict]:
        p = self.raw_path
        if p.is_dir():
            files = list(p.glob("*.json")) + list(p.glob("*.jsonl")) + list(p.glob("*.csv"))
        else:
            files = [p]
        for fp in files:
            if str(fp).endswith(".csv"):
                with open(fp, encoding="utf-8", newline="") as f:
                    for row in csv.DictReader(f):
                        yield row
            else:  # .json / .jsonl — TESORO ships JSONL (one object per line)
                text = open(fp, encoding="utf-8").read().strip()
                for line in text.splitlines():
                    line = line.strip()
                    if line:
                        try:
                            yield json.loads(line)
                        except json.JSONDecodeError:
                            continue

    def _to_pair(self, raw: dict) -> DocPair | None:
        code = raw.get("code") or raw.get("snippet") or raw.get("method", "")
        doc  = raw.get("comment") or raw.get("text", "")
        if not code or not doc:
            return None
        pair = DocPair(self.name, TASK_SATD, "java", code, doc,
                       repo=str(raw.get("id", "tesoro")))
        pair.meta["satd_label"] = raw.get("classification") or raw.get("label")
        return pair
