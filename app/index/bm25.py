from __future__ import annotations

import math
import pickle
import re
from dataclasses import dataclass
from typing import Dict, List


_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> List[str]:
    text = (text or "").lower()
    return _TOKEN_RE.findall(text)


@dataclass
class BM25Index:
    """
    Minimal, stable BM25 implementation that is:
    - pickleable (no lambdas / local classes)
    - deterministic
    - dependency-light (no rank_bm25 / sklearn)
    """
    k1: float
    b: float
    docs_len: List[int]
    avgdl: float
    df: Dict[str, int]                # document frequency
    idf: Dict[str, float]             # idf per term
    tf: List[Dict[str, int]]          # per-document term frequency maps

    @classmethod
    def build(cls, texts: List[str], k1: float = 1.5, b: float = 0.75) -> "BM25Index":
        tf: List[Dict[str, int]] = []
        df: Dict[str, int] = {}
        docs_len: List[int] = []

        for t in texts:
            terms = _tokenize(t)
            docs_len.append(len(terms))

            freq: Dict[str, int] = {}
            for w in terms:
                freq[w] = freq.get(w, 0) + 1
            tf.append(freq)

            for w in freq.keys():
                df[w] = df.get(w, 0) + 1

        n_docs = max(1, len(texts))
        avgdl = sum(docs_len) / n_docs if n_docs else 0.0

        idf: Dict[str, float] = {}
        for w, f in df.items():
            # classic BM25 idf
            idf[w] = math.log(1.0 + (n_docs - f + 0.5) / (f + 0.5))

        return cls(
            k1=k1,
            b=b,
            docs_len=docs_len,
            avgdl=avgdl,
            df=df,
            idf=idf,
            tf=tf,
        )

    def scores(self, query: str) -> List[float]:
        q_terms = _tokenize(query)
        if not q_terms:
            return [0.0] * len(self.tf)

        scores = [0.0] * len(self.tf)
        for i, doc_tf in enumerate(self.tf):
            dl = self.docs_len[i] if i < len(self.docs_len) else 0
            denom_norm = (1.0 - self.b) + self.b * (dl / self.avgdl if self.avgdl > 0 else 0.0)

            s = 0.0
            for w in q_terms:
                if w not in doc_tf:
                    continue
                f = doc_tf[w]
                idf = self.idf.get(w, 0.0)
                numer = f * (self.k1 + 1.0)
                denom = f + self.k1 * denom_norm
                s += idf * (numer / denom)
            scores[i] = s

        return scores


def save_bm25(index: BM25Index, path: str) -> None:
    with open(path, "wb") as f:
        pickle.dump(index, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_bm25(path: str) -> BM25Index:
    with open(path, "rb") as f:
        obj = pickle.load(f)
    if not isinstance(obj, BM25Index):
        raise TypeError(f"BM25 pickle is not a BM25Index. Got: {type(obj)}")
    return obj
