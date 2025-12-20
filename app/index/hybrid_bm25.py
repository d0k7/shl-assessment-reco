from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Tuple

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+", re.UNICODE)

def tokenize(text: str) -> List[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text or "") if len(t) >= 2]


@dataclass
class BM25:
    """
    Minimal BM25 implementation (no external deps).
    Stores doc term-frequencies + idf.
    """
    k1: float = 1.5
    b: float = 0.75
    doc_freq: Dict[str, int] = None
    idf: Dict[str, float] = None
    doc_len: List[int] = None
    avgdl: float = 0.0
    tf: List[Counter] = None
    n_docs: int = 0

    @classmethod
    def build(cls, docs: List[str], k1: float = 1.5, b: float = 0.75) -> "BM25":
        tf: List[Counter] = []
        doc_freq: Dict[str, int] = {}
        doc_len: List[int] = []

        for d in docs:
            toks = tokenize(d)
            c = Counter(toks)
            tf.append(c)
            doc_len.append(len(toks))
            for term in c.keys():
                doc_freq[term] = doc_freq.get(term, 0) + 1

        n_docs = len(docs)
        avgdl = sum(doc_len) / max(1, n_docs)

        # BM25 idf with +1 smoothing
        idf: Dict[str, float] = {}
        for term, df in doc_freq.items():
            idf[term] = math.log(1 + (n_docs - df + 0.5) / (df + 0.5))

        return cls(k1=k1, b=b, doc_freq=doc_freq, idf=idf, doc_len=doc_len, avgdl=avgdl, tf=tf, n_docs=n_docs)

    def score(self, query: str, top_n: int = 100) -> List[Tuple[int, float]]:
        q_terms = tokenize(query)
        if not q_terms:
            return []

        scores: List[float] = [0.0] * self.n_docs

        for i in range(self.n_docs):
            dl = self.doc_len[i]
            denom_norm = self.k1 * (1 - self.b + self.b * (dl / (self.avgdl or 1.0)))
            doc_tf = self.tf[i]

            s = 0.0
            for t in q_terms:
                if t not in doc_tf:
                    continue
                f = doc_tf[t]
                idf = self.idf.get(t, 0.0)
                s += idf * (f * (self.k1 + 1)) / (f + denom_norm)
            scores[i] = s

        # return top_n indices by score
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return [(i, sc) for i, sc in ranked[:top_n] if sc > 0]
