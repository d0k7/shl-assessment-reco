from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import faiss  # type: ignore
import numpy as np
from sentence_transformers import SentenceTransformer

from app.schemas.catalog import CatalogItem


# ----------------------------
# Canonicalization helpers
# ----------------------------
_SLUG_RE = re.compile(r"/product-catalog/view/([^/?#]+)", re.IGNORECASE)


def slug_from_url(url: str) -> Optional[str]:
    m = _SLUG_RE.search(url)
    if not m:
        return None
    return m.group(1).strip().strip("/").lower()


def canonicalize_url_for_output(url: str) -> str:
    """
    Output URLs in a consistent format that matches the dataset better.
    Dataset mostly uses: https://www.shl.com/solutions/products/product-catalog/view/<slug>
    """
    slug = slug_from_url(url)
    if not slug:
        return url.rstrip("/")
    return f"https://www.shl.com/solutions/products/product-catalog/view/{slug}"


def normalize_text(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def clean_query(q: str) -> str:
    """
    Clean long JD text a bit so embeddings + lexical both behave better.
    """
    q = q.strip()
    q = re.sub(r"\s+", " ", q)
    # remove extremely repetitive punctuation blocks
    q = re.sub(r"[_\-]{4,}", " ", q)
    return q.strip()


# ----------------------------
# Lightweight TF-IDF (no sklearn dependency)
# ----------------------------
def _tokenize(text: str) -> List[str]:
    text = normalize_text(text)
    # keep words+numbers, drop tiny tokens
    toks = re.findall(r"[a-z0-9]+", text)
    return [t for t in toks if len(t) >= 2]


@dataclass
class TfidfIndex:
    vocab: Dict[str, int]
    idf: np.ndarray  # shape (V,)
    doc_vecs: np.ndarray  # shape (N, V) sparse-ish dense float32 (small corpus is ok)

    @staticmethod
    def build(docs: List[str]) -> "TfidfIndex":
        # DF
        df: Dict[str, int] = {}
        tokenized = [_tokenize(d) for d in docs]
        for toks in tokenized:
            for t in set(toks):
                df[t] = df.get(t, 0) + 1

        vocab = {t: i for i, t in enumerate(df.keys())}
        N = len(docs)
        V = len(vocab)

        idf = np.zeros((V,), dtype=np.float32)
        for t, i in vocab.items():
            # smooth IDF
            idf[i] = np.log((N + 1) / (df[t] + 1)) + 1.0

        # TF-IDF vectors (dense; V may be few 10k max; manageable for ~500 docs)
        doc_vecs = np.zeros((N, V), dtype=np.float32)
        for di, toks in enumerate(tokenized):
            if not toks:
                continue
            tf: Dict[int, int] = {}
            for t in toks:
                idx = vocab.get(t)
                if idx is not None:
                    tf[idx] = tf.get(idx, 0) + 1
            # tf-idf
            for idx, c in tf.items():
                doc_vecs[di, idx] = float(c) * idf[idx]

            # L2 normalize
            norm = np.linalg.norm(doc_vecs[di])
            if norm > 0:
                doc_vecs[di] /= norm

        return TfidfIndex(vocab=vocab, idf=idf, doc_vecs=doc_vecs)

    def encode(self, text: str) -> np.ndarray:
        toks = _tokenize(text)
        V = len(self.vocab)
        vec = np.zeros((V,), dtype=np.float32)
        if not toks:
            return vec
        tf: Dict[int, int] = {}
        for t in toks:
            idx = self.vocab.get(t)
            if idx is not None:
                tf[idx] = tf.get(idx, 0) + 1
        for idx, c in tf.items():
            vec[idx] = float(c) * self.idf[idx]
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec


# ----------------------------
# Query intent heuristics for balancing
# ----------------------------
_TECH_HINTS = {
    "java", "python", "sql", "javascript", "node", "react", "c++", "c#", "aws", "azure",
    "kubernetes", "docker", "etl", "data", "ml", "devops", "backend", "frontend",
}
_SOFT_HINTS = {
    "collaborat", "stakeholder", "communication", "team", "leadership", "culture",
    "manage", "influence", "customer", "sales", "service", "empathy",
}


def needs_balance(query: str) -> bool:
    q = normalize_text(query)
    tech = any(t in q for t in _TECH_HINTS)
    soft = any(t in q for t in _SOFT_HINTS)
    return tech and soft


def is_knowledge_skill(item: CatalogItem) -> bool:
    # K appears in many variants like ["K REMOTE TESTING"] in your current crawl
    tt = " ".join(item.test_type or []).upper()
    return " K" in f" {tt}" or tt.startswith("K")


def is_personality_behavior(item: CatalogItem) -> bool:
    tt = " ".join(item.test_type or []).upper()
    return " P" in f" {tt}" or tt.startswith("P")


# ----------------------------
# Recommender
# ----------------------------
class SHLRecommender:
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        index_path: str = "data/index/catalog.faiss",
        meta_path: str = "data/index/catalog_meta.jsonl",
    ) -> None:
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.index = faiss.read_index(index_path)
        self.items = self._load_meta(meta_path)

        # Prepare TF-IDF docs for hybrid retrieval
        docs = [self._doc_text(it) for it in self.items]
        self.tfidf = TfidfIndex.build(docs)

    @staticmethod
    def _load_meta(path: str) -> List[CatalogItem]:
        items: List[CatalogItem] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                items.append(CatalogItem(**json.loads(line)))
        return items

    @staticmethod
    def _doc_text(it: CatalogItem) -> str:
        # same spirit as build_text: include fields that help matching
        parts = [
            f"Name: {it.name}",
            f"Description: {it.description or ''}",
            f"TestType: {' '.join(it.test_type or [])}",
            f"Duration: {it.duration}",
            f"Remote: {it.remote_support}",
            f"Adaptive: {it.adaptive_support}",
        ]
        return "\n".join(parts)

    def _encode_query(self, query: str) -> np.ndarray:
        vec = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        return vec.astype(np.float32, copy=False)

    def recommend(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Hybrid retrieval:
          1) semantic candidates via FAISS (top 80)
          2) lexical candidates via TF-IDF cosine (top 80)
          3) merge, rerank by weighted score
          4) enforce balance if query spans domains (K + P)
        """
        query = clean_query(query)

        # ---- semantic candidates
        q_emb = self._encode_query(query)
        sem_k = min(80, max(top_k, 10) * 8)
        sem_scores, sem_idxs = self.index.search(q_emb, sem_k)

        sem_scores = sem_scores[0].tolist()
        sem_idxs = sem_idxs[0].tolist()

        # ---- lexical candidates (cosine of normalized tfidf vectors)
        q_tfidf = self.tfidf.encode(query)
        # if query has no vocab overlap, lexical score will be zero
        lex_scores_all = self.tfidf.doc_vecs @ q_tfidf
        lex_k = min(80, max(top_k, 10) * 8)
        lex_idxs = np.argsort(-lex_scores_all)[:lex_k].tolist()

        # ---- merge candidates
        cand: Dict[int, Dict[str, float]] = {}
        for s, i in zip(sem_scores, sem_idxs):
            if i < 0:
                continue
            cand.setdefault(i, {})
            cand[i]["sem"] = float(s)

        for i in lex_idxs:
            cand.setdefault(i, {})
            cand[i]["lex"] = float(lex_scores_all[i])

        # ---- rerank (weights can be tuned)
        # semantic is cosine (0..1-ish), lexical is cosine (0..1)
        # add small boost if query mentions duration constraints and item has duration
        q_norm = normalize_text(query)
        wants_hour = ("hour" in q_norm) or ("60" in q_norm) or ("1 hour" in q_norm)

        scored: List[Tuple[int, float]] = []
        for i, d in cand.items():
            sem = d.get("sem", 0.0)
            lex = d.get("lex", 0.0)
            score = 0.70 * sem + 0.30 * lex

            if wants_hour and (self.items[i].duration and self.items[i].duration >= 40):
                score += 0.02

            scored.append((i, score))

        scored.sort(key=lambda x: x[1], reverse=True)

        # ---- pick final with balance if needed
        final_idxs: List[int] = []
        if needs_balance(query):
            # take from top 80, ensure at least 2 K and 2 P if possible
            want_k = max(1, top_k // 3)
            want_p = max(1, top_k // 3)

            picked_k, picked_p = 0, 0
            # first pass: satisfy quotas
            for i, _ in scored:
                it = self.items[i]
                if i in final_idxs:
                    continue
                if picked_k < want_k and is_knowledge_skill(it):
                    final_idxs.append(i)
                    picked_k += 1
                elif picked_p < want_p and is_personality_behavior(it):
                    final_idxs.append(i)
                    picked_p += 1
                if len(final_idxs) >= top_k:
                    break

            # second pass: fill remaining by best score
            if len(final_idxs) < top_k:
                for i, _ in scored:
                    if i not in final_idxs:
                        final_idxs.append(i)
                    if len(final_idxs) >= top_k:
                        break
        else:
            final_idxs = [i for i, _ in scored[:top_k]]

        # ---- build response
        out: List[Dict[str, Any]] = []
        for i in final_idxs:
            it = self.items[i]
            out.append(
                {
                    "name": it.name,
                    "url": canonicalize_url_for_output(str(it.url)),
                    "description": it.description or "",
                    "duration": int(it.duration or 0),
                    "remote_support": it.remote_support or "No",
                    "adaptive_support": it.adaptive_support or "No",
                    "test_type": list(it.test_type or []),
                }
            )
        return out
