"""
loaders/funcom.py — Funcom (LeClair et al., 2019). Java method -> comment summary.
Real format (from exploration): two parallel files aligned line-by-line:
    dats.train  (code, one method per line, tokenized)
    coms.train  (comment/summary, one per line)
Files may sit in data/raw/funcom/ OR directly in data/raw/. Also accepts a single
funcom.jsonl with {code, comment}.
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Iterator
from .base import BaseLoader
from schema import DocPair, TASK_SUMMARIZATION


class FuncomLoader(BaseLoader):
    name = "funcom"

    def _candidate_dirs(self):
        dirs = []
        if self.raw_path.is_dir():
            dirs.append(self.raw_path)
        # Funcom files are often dropped straight into data/raw/, so also look one up.
        if self.raw_path.name == "funcom":
            dirs.append(self.raw_path.parent)
        return dirs

    def _find_pair_files(self):
        for d in self._candidate_dirs():
            fcode = next(d.glob("dats.*"), None) or next(d.glob("*function*"), None)
            fcom  = next(d.glob("coms.*"), None) or next(d.glob("*comment*"), None)
            if fcode and fcom:
                return fcode, fcom
        return None, None

    def _find_jsonl(self):
        if str(self.raw_path).endswith(".jsonl") and self.raw_path.exists():
            return self.raw_path
        for d in self._candidate_dirs():
            j = next(d.glob("funcom*.jsonl"), None)
            if j:
                return j
        return None

    def _iter_raw(self) -> Iterator[dict]:
        jsonl = self._find_jsonl()
        if jsonl:
            with open(jsonl, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        yield json.loads(line)
            return
        fcode, fcom = self._find_pair_files()
        if fcode and fcom:
            with open(fcode, encoding="utf-8") as a, open(fcom, encoding="utf-8") as b:
                for code, com in zip(a, b):
                    yield {"code": code.rstrip("\n"), "comment": com.rstrip("\n")}

    def _to_pair(self, raw: dict) -> DocPair | None:
        code = raw.get("code") or raw.get("function") or raw.get("src", "")
        doc  = raw.get("comment") or raw.get("summary") or raw.get("nl", "")
        if not code or not doc:
            return None
        return DocPair(self.name, TASK_SUMMARIZATION, "java", code, doc,
                       repo=raw.get("repo", "funcom"))

    def available(self) -> bool:
        if self._find_jsonl():
            return True
        fcode, fcom = self._find_pair_files()
        return bool(fcode and fcom)
