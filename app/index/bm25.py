from __future__ import annotations

import math
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np


_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[._+-][A-Za-z0-9]+)*")


def tokenize(text: str) -> List[str]:
    text = (text or "").lower()
    return _TOKEN_RE.findall(text)


@dataclass
class BM25Index:
    """
    Lightweight BM25 implementation with only primitive, pickle-safe state.
    This class MUST live in a stable import path: app.index.bm25.BM25Index
    """
    doc_freq: Dict[str, int]
    idf: Dict[str, float]
    doc_len: List[int]
    avgdl: float
    tf: List[Dict[str, int]]
    k1: float = 1.5
    b: float = 0.75

    @classmethod
    def build(cls, docs: Sequence[str], k1: float = 1.5, b: float = 0.75) -> "BM25Index":
        tf_list: List[Dict[str, int]] = []
        doc_freq: Dict[str, int] = {}
        doc_len: List[int] = []

        for doc in docs:
            toks = tokenize(doc)
            doc_len.append(len(toks))
            freqs: Dict[str, int] = {}
            for t in toks:
                freqs[t] = freqs.get(t, 0) + 1
            tf_list.append(freqs)

            for t in freqs.keys():
                doc_freq[t] = doc_freq.get(t, 0) + 1

        n_docs = len(docs)
        avgdl = float(sum(doc_len) / max(1, n_docs))

        # BM25 idf (with +1 to keep positive)
        idf: Dict[str, float] = {}
        for term, df in doc_freq.items():
            idf[term] = math.log((n_docs - df + 0.5) / (df + 0.5) + 1.0)

        return cls(
            doc_freq=doc_freq,
            idf=idf,
            doc_len=doc_len,
            avgdl=avgdl,
            tf=tf_list,
            k1=k1,
            b=b,
        )

    def score(self, query: str) -> np.ndarray:
        q_terms = tokenize(query)
        if not q_terms:
            return np.zeros(len(self.tf), dtype=np.float32)

        scores = np.zeros(len(self.tf), dtype=np.float32)

        for i, freqs in enumerate(self.tf):
            dl = self.doc_len[i]
            denom_norm = self.k1 * (1 - self.b + self.b * (dl / max(1e-9, self.avgdl)))

            s = 0.0
            for t in q_terms:
                if t not in freqs:
                    continue
                tf = freqs[t]
                term_idf = self.idf.get(t, 0.0)
                s += term_idf * (tf * (self.k1 + 1.0)) / (tf + denom_norm)
            scores[i] = float(s)

        return scores

    def top_n(self, query: str, n: int) -> Tuple[np.ndarray, np.ndarray]:
        scores = self.score(query)
        if n <= 0:
            return np.array([], dtype=np.int64), np.array([], dtype=np.float32)
        n = min(n, len(scores))
        idx = np.argpartition(-scores, n - 1)[:n]
        idx = idx[np.argsort(-scores[idx])]
        return idx.astype(np.int64), scores[idx].astype(np.float32)


def save_bm25(index: BM25Index, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(index, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_bm25(path: str) -> BM25Index:
    with open(path, "rb") as f:
        obj = pickle.load(f)
    if not isinstance(obj, BM25Index):
        raise TypeError(f"BM25 pickle is not BM25Index. Got: {type(obj)}")
    return obj
