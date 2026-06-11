"""
loaders/base.py — every dataset adapter subclasses this and returns DocPairs.
The framework never touches a dataset's raw format directly; only loaders do.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterator, List
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))
from schema import DocPair, is_valid_pair


class BaseLoader(ABC):
    name: str = "base"
    default_task: str = "doc_generation"
    language: str = "java"

    def __init__(self, raw_path: Path, language: str | None = None):
        self.raw_path = Path(raw_path)
        if language:
            self.language = language

    @abstractmethod
    def _iter_raw(self) -> Iterator[dict]:
        """Yield raw dataset records (loader-specific). Override per dataset."""
        ...

    @abstractmethod
    def _to_pair(self, raw: dict) -> DocPair | None:
        """Map ONE raw record to a DocPair, or None if unusable."""
        ...

    def load(self, limit: int | None = None) -> List[DocPair]:
        pairs: list[DocPair] = []
        for raw in self._iter_raw():
            try:
                p = self._to_pair(raw)
            except Exception:
                p = None
            if p and is_valid_pair(p.code_unit, p.reference_doc):
                pairs.append(p)
                if limit and len(pairs) >= limit:
                    break
        return pairs

    def available(self) -> bool:
        return self.raw_path.exists()
