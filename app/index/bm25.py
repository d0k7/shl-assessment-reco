from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

from rank_bm25 import BM25Okapi  # lightweight, no torch


_TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)

# small, practical “skills” list to boost obvious intent
_SKILL_HINTS = {
    "java",
    "python",
    "sql",
    "javascript",
    "react",
    "node",
    "csharp",
    "c#",
    "dotnet",
    ".net",
    "aws",
    "azure",
    "gcp",
    "kubernetes",
    "docker",
    "spring",
    "hibernate",
    "microservices",
    "jira",
    "salesforce",
    "sap",
    "informatica",
    "etl",
    "devops",
}


def tokenize(text: str) -> List[str]:
    return [m.group(0).lower() for m in _TOKEN_RE.finditer(text or "")]


@dataclass(frozen=True)
class BM25Store:
    """
    Two-channel BM25:
      - title/index on name (high weight)
      - body/index on description + misc (lower weight)
    """
    bm25_title: BM25Okapi
    bm25_body: BM25Okapi
    title_tokens: List[List[str]]
    body_tokens: List[List[str]]

    @staticmethod
    def build(title_texts: Sequence[str], body_texts: Sequence[str]) -> "BM25Store":
        title_tokens = [tokenize(t) for t in title_texts]
        body_tokens = [tokenize(t) for t in body_texts]
        return BM25Store(
            bm25_title=BM25Okapi(title_tokens),
            bm25_body=BM25Okapi(body_tokens),
            title_tokens=title_tokens,
            body_tokens=body_tokens,
        )

    def topk(
        self,
        query: str,
        k: int,
        *,
        title_weight: float = 2.5,
        body_weight: float = 1.0,
        skill_boost: float = 1.5,
        skill_boost_max: float = 3.0,
        title_exact_boost: float = 2.0,
        titles_raw: Sequence[str] | None = None,
    ) -> List[Tuple[int, float]]:
        """
        Returns: list of (doc_index, score) sorted desc.
        """
        q_tokens = tokenize(query)
        if not q_tokens:
            return []

        # BM25Okapi returns numpy arrays; treat as sequences
        s_title = self.bm25_title.get_scores(q_tokens)
        s_body = self.bm25_body.get_scores(q_tokens)

        # combine
        scores = (title_weight * s_title) + (body_weight * s_body)

        # Skill boosting:
        # If query contains "java", boost items whose title contains "java".
        q_skills = {t for t in q_tokens if t in _SKILL_HINTS}
        if q_skills and titles_raw is not None:
            for i, title in enumerate(titles_raw):
                tl = (title or "").lower()
                matched = 0
                for sk in q_skills:
                    if sk in tl:
                        matched += 1
                if matched:
                    # cap multiplicative boost
                    mult = min(skill_boost_max, (skill_boost ** matched))
                    scores[i] = scores[i] * mult

        # Extra exact-title token boost (helps "Java 8", "Core Java", etc.)
        if titles_raw is not None:
            q_set = set(q_tokens)
            for i, title in enumerate(titles_raw):
                t_tokens = set(tokenize(title))
                if t_tokens and (len(q_set & t_tokens) >= 2):
                    scores[i] = scores[i] * title_exact_boost

        # top-k selection
        # (avoid importing numpy explicitly; keep it simple)
        scored = list(enumerate(scores))
        scored.sort(key=lambda x: float(x[1]), reverse=True)
        return [(idx, float(sc)) for idx, sc in scored[:k]]
