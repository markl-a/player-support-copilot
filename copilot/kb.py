"""Policy search over kb/*.md. BM25 on character bigrams; each bullet is one retrievable unit, so a reply cites
the exact rule it relies on (file + line)."""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

KB = Path(__file__).resolve().parent.parent / "kb"


@dataclass(frozen=True)
class Passage:
    file: str
    line: int
    title: str
    text: str

    @property
    def cite(self) -> str:
        return f"{self.file}:{self.line}"


def bigrams(s: str) -> list[str]:
    s = re.sub(r"\s+", "", s.lower())
    return [s[i:i + 2] for i in range(len(s) - 1)]


class Index:
    def __init__(self, root: Path = KB, k1: float = 1.2, b: float = 0.75):
        self.docs: list[Passage] = []
        for f in sorted(root.glob("*.md")):
            lines = f.read_text().splitlines()
            title = lines[0].lstrip("# ").strip()
            for i, l in enumerate(lines, 1):
                if l.startswith("- "):
                    self.docs.append(Passage(f.name, i, title, l[2:].strip()))
        self.tfs = [Counter(bigrams(p.title + p.text)) for p in self.docs]
        self.avg = sum(sum(t.values()) for t in self.tfs) / len(self.tfs)
        df = Counter(g for t in self.tfs for g in t)
        n = len(self.docs)
        self.idf = {g: math.log(1 + (n - c + 0.5) / (c + 0.5)) for g, c in df.items()}
        self.k1, self.b = k1, b

    def search(self, query: str, k: int = 3, file: str | None = None) -> list[Passage]:
        q = bigrams(query)
        scored = []
        for p, tf in zip(self.docs, self.tfs):
            if file and p.file != file:
                continue
            L = sum(tf.values())
            s = sum(self.idf.get(g, 0) * tf[g] * (self.k1 + 1) / (tf[g] + self.k1 * (1 - self.b + self.b * L / self.avg)) for g in q if g in tf)
            scored.append((s, p))
        return [p for s, p in sorted(scored, key=lambda x: -x[0])[:k] if s > 0]
