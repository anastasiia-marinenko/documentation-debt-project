"""
loaders/robustness_copilot.py — Mastropaolo et al. (2022) robustness dataset.
Real schema (from exploration): robustness_copilot.csv, 892 Java methods, 33 projects.
    body                 -> method source code
    javaDoc              -> full reference Javadoc  (javaDocFirstSentence = summary)
    project, methodName  -> provenance
    pegasusPerturbed / pivotingPerturbed -> paraphrased NL variants (stored in meta)
"""
from __future__ import annotations
import csv, json
from pathlib import Path
from typing import Iterator
from .base import BaseLoader
from schema import DocPair, TASK_DOC_GENERATION


class RobustnessCopilotLoader(BaseLoader):
    name = "robustness_copilot"

    def _iter_raw(self) -> Iterator[dict]:
        p = self.raw_path
        files = (list(p.glob("*.csv")) + list(p.glob("*.jsonl"))) if p.is_dir() else [p]
        for fp in files:
            if str(fp).endswith(".jsonl"):
                with open(fp, encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            yield json.loads(line)
            else:  # csv
                with open(fp, encoding="utf-8", newline="") as f:
                    for row in csv.DictReader(f):
                        yield row

    def _to_pair(self, raw: dict) -> DocPair | None:
        code = raw.get("body") or raw.get("code") or raw.get("method", "")
        doc  = raw.get("javaDoc") or raw.get("javaDocFirstSentence") or raw.get("comment", "")
        if not code or not doc:
            return None
        pair = DocPair(self.name, TASK_DOC_GENERATION, "java", code, doc,
                       repo=raw.get("project", "robustness_copilot"))
        pair.meta["method_name"] = raw.get("methodName")
        for k in ("pegasusPerturbed", "pivotingPerturbed"):
            if raw.get(k):
                pair.meta[k] = raw[k]
        return pair
