"""Small local BM25 ranker over item text."""

from __future__ import annotations

import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

from llm4rec.data.text_fields import item_text
from llm4rec.io.artifacts import write_json
from llm4rec.rankers.base import RankingExample, RankingResult, result_from_scores

TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Tokenize item/user text for dependency-free BM25."""

    return TOKEN_RE.findall(text.lower())


class BM25Ranker:
    """Rank candidate item texts against a user-history text query."""

    name = "bm25"

    def __init__(self, *, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = float(k1)
        self.b = float(b)
        self.item_texts: dict[str, str] = {}
        self.doc_tokens: dict[str, list[str]] = {}
        self.doc_tf: dict[str, Counter[str]] = {}
        self.document_frequency: Counter[str] = Counter()
        self.average_doc_length = 0.0
        self.num_documents = 0

    def fit(
        self,
        train_interactions: list[dict[str, Any]],
        item_records: list[dict[str, Any]],
    ) -> None:
        self.item_texts = {str(row["item_id"]): item_text(row) for row in item_records}
        self.doc_tokens = {item_id: tokenize(text) for item_id, text in self.item_texts.items()}
        self.doc_tf = {item_id: Counter(tokens) for item_id, tokens in self.doc_tokens.items()}
        self.document_frequency = Counter()
        for tokens in self.doc_tokens.values():
            self.document_frequency.update(set(tokens))
        self.num_documents = len(self.doc_tokens)
        total_len = sum(len(tokens) for tokens in self.doc_tokens.values())
        self.average_doc_length = total_len / float(self.num_documents or 1)

    def rank(self, example: RankingExample) -> RankingResult:
        query_tokens = self._query_tokens(example.history)
        scores = {str(item_id): self.score_item(str(item_id), query_tokens) for item_id in example.candidate_items}
        return result_from_scores(
            example=example,
            scores_by_item=scores,
            metadata={
                "query_history_length": len(example.history),
                "scorer": "local_bm25",
                "k1": self.k1,
                "b": self.b,
            },
        )

    def score_item(self, item_id: str, query_tokens: list[str]) -> float:
        tokens = self.doc_tokens.get(item_id, [])
        if not tokens or not query_tokens:
            return 0.0
        tf = self.doc_tf.get(item_id, Counter())
        doc_len = len(tokens)
        score = 0.0
        for token, query_count in Counter(query_tokens).items():
            freq = tf.get(token, 0)
            if freq == 0:
                continue
            df = self.document_frequency.get(token, 0)
            idf = math.log(1.0 + (self.num_documents - df + 0.5) / (df + 0.5))
            denom = freq + self.k1 * (1.0 - self.b + self.b * doc_len / (self.average_doc_length or 1.0))
            score += float(query_count) * idf * (freq * (self.k1 + 1.0)) / denom
        return score

    def save_artifact(self, output_dir: str | Path) -> None:
        write_json(
            Path(output_dir) / "bm25_metadata.json",
            {
                "average_doc_length": self.average_doc_length,
                "b": self.b,
                "k1": self.k1,
                "num_documents": self.num_documents,
                "vocabulary_size": len(self.document_frequency),
            },
        )

    def _query_tokens(self, history: list[str]) -> list[str]:
        tokens: list[str] = []
        for item_id in history:
            tokens.extend(tokenize(self.item_texts.get(str(item_id), "")))
        return tokens
