"""Vectorized BM25 scorer for compact shared-pool candidate protocols."""

from __future__ import annotations

import math
from collections import Counter
from typing import Any

from llm4rec.data.text_fields import item_text
from llm4rec.rankers.bm25 import tokenize
from llm4rec.scoring.candidate_batch import CandidateBatch


class BM25SharedPoolScorer:
    """Score shared-pool candidate matrices with a train-visible item BM25 matrix."""

    name = "bm25_shared_pool"

    def __init__(self, *, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = float(k1)
        self.b = float(b)
        self.item_texts: dict[str, str] = {}
        self.item_ids: list[str] = []
        self.item_to_idx: dict[str, int] = {}
        self.vocab: dict[str, int] = {}
        self.vectorizer_type = "local_inverted_bm25"
        self.item_matrix: Any | None = None
        self.inverted: dict[str, list[tuple[str, float]]] = {}
        self.metadata: dict[str, Any] = {}

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> "BM25SharedPoolScorer":
        """Adapt the legacy paper-matrix BM25 state for shared-pool scoring."""

        scorer = cls(
            k1=float(state.get("metadata", {}).get("k1", 1.5)),
            b=float(state.get("metadata", {}).get("b", 0.75)),
        )
        scorer.item_texts = {str(key): str(value) for key, value in state.get("item_texts", {}).items()}
        scorer.inverted = dict(state.get("inverted", {}))
        scorer.metadata = dict(state.get("metadata", {}))
        scorer.metadata["vectorizer_type"] = scorer.vectorizer_type
        return scorer

    def fit(self, train_interactions: list[dict[str, Any]], item_records: list[dict[str, Any]]) -> None:
        """Build a sparse BM25 item-term matrix from item records."""

        del train_interactions
        self.item_texts = {str(row["item_id"]): item_text(row) for row in item_records}
        self.item_ids = [str(row["item_id"]) for row in item_records]
        self.item_to_idx = {item_id: index for index, item_id in enumerate(self.item_ids)}
        tokenized = {item_id: tokenize(self.item_texts.get(item_id, "")) for item_id in self.item_ids}
        document_frequency: Counter[str] = Counter()
        doc_tf: dict[str, Counter[str]] = {}
        doc_lengths: dict[str, int] = {}
        total_len = 0
        for item_id, tokens in tokenized.items():
            tf = Counter(tokens)
            doc_tf[item_id] = tf
            doc_lengths[item_id] = len(tokens)
            total_len += len(tokens)
            document_frequency.update(set(tf))
        self.vocab = {token: index for index, token in enumerate(sorted(document_frequency))}
        num_documents = len(self.item_ids)
        average_doc_length = total_len / float(num_documents or 1)
        try:
            import numpy as np
            from scipy import sparse

            rows: list[int] = []
            cols: list[int] = []
            data: list[float] = []
            for item_id in self.item_ids:
                doc_len = doc_lengths.get(item_id, 0)
                if doc_len <= 0:
                    continue
                row_index = self.item_to_idx[item_id]
                for token, freq in doc_tf[item_id].items():
                    df = document_frequency.get(token, 0)
                    idf = math.log(1.0 + (num_documents - df + 0.5) / (df + 0.5))
                    denom = freq + self.k1 * (1.0 - self.b + self.b * doc_len / (average_doc_length or 1.0))
                    rows.append(row_index)
                    cols.append(self.vocab[token])
                    data.append(float(idf * (freq * (self.k1 + 1.0)) / denom))
            self.item_matrix = sparse.csr_matrix(
                (np.asarray(data, dtype=np.float32), (rows, cols)),
                shape=(len(self.item_ids), len(self.vocab)),
                dtype=np.float32,
            )
            self.vectorizer_type = "scipy_sparse_bm25"
        except ModuleNotFoundError:
            self.item_matrix = None
            inverted: dict[str, list[tuple[str, float]]] = {}
            for item_id, tf in doc_tf.items():
                doc_len = doc_lengths.get(item_id, 0)
                if doc_len <= 0:
                    continue
                for token, freq in tf.items():
                    df = document_frequency.get(token, 0)
                    idf = math.log(1.0 + (num_documents - df + 0.5) / (df + 0.5))
                    denom = freq + self.k1 * (1.0 - self.b + self.b * doc_len / (average_doc_length or 1.0))
                    inverted.setdefault(token, []).append(
                        (item_id, float(idf * (freq * (self.k1 + 1.0)) / denom))
                    )
            self.inverted = inverted
            self.vectorizer_type = "local_inverted_bm25"
        self.metadata = {
            "average_doc_length": average_doc_length,
            "b": self.b,
            "candidate_pool_size": None,
            "k1": self.k1,
            "num_documents": num_documents,
            "vectorizer_type": self.vectorizer_type,
            "vocabulary_size": len(self.vocab) if self.vocab else len(document_frequency),
        }

    def score_batch(self, batch: CandidateBatch) -> Any:
        """Return a dense [B, C] score matrix for one candidate batch."""

        if self.item_matrix is None:
            return self._score_batch_inverted(batch)
        import numpy as np
        from scipy import sparse

        query = self._query_matrix(batch.histories)
        score_matrix = np.zeros((batch.batch_size, batch.candidate_size), dtype=np.float32)
        if query.nnz == 0:
            return score_matrix
        unique_items = sorted({item for row in batch.candidate_item_ids for item in row if item in self.item_to_idx})
        if not unique_items:
            return score_matrix
        unique_indices = [self.item_to_idx[item] for item in unique_items]
        subset = self.item_matrix[unique_indices]
        dense_scores = (query @ subset.T).toarray()
        unique_pos = {item: index for index, item in enumerate(unique_items)}
        candidate_pos = np.asarray(
            [[unique_pos.get(item, -1) for item in row] for row in batch.candidate_item_ids],
            dtype=np.int32,
        )
        valid = candidate_pos >= 0
        row_indices = np.arange(batch.batch_size, dtype=np.int32)[:, None]
        score_matrix[valid] = dense_scores[row_indices, candidate_pos][valid]
        return score_matrix

    def _query_matrix(self, histories: list[list[str]]) -> Any:
        import numpy as np
        from scipy import sparse

        rows: list[int] = []
        cols: list[int] = []
        data: list[float] = []
        for row_index, history in enumerate(histories):
            counts: Counter[str] = Counter()
            for item_id in history:
                counts.update(tokenize(self.item_texts.get(str(item_id), "")))
            for token, count in counts.items():
                col = self.vocab.get(token)
                if col is None:
                    continue
                rows.append(row_index)
                cols.append(col)
                data.append(float(count))
        return sparse.csr_matrix(
            (np.asarray(data, dtype=np.float32), (rows, cols)),
            shape=(len(histories), len(self.vocab)),
            dtype=np.float32,
        )

    def _score_batch_inverted(self, batch: CandidateBatch) -> list[list[float]]:
        output: list[list[float]] = []
        for history, candidates in zip(batch.histories, batch.candidate_item_ids):
            query_tokens: list[str] = []
            for item_id in history:
                query_tokens.extend(tokenize(self.item_texts.get(str(item_id), "")))
            query_counts = Counter(query_tokens)
            candidate_set = {str(item) for item in candidates}
            scores = {str(item): 0.0 for item in candidates}
            for token, count in query_counts.items():
                for item_id, contribution in self.inverted.get(token, []):
                    if item_id in candidate_set:
                        scores[item_id] += float(count) * float(contribution)
            output.append([float(scores[str(item)]) for item in candidates])
        return output
