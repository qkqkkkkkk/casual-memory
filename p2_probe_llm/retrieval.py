from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any

import numpy as np


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", str(text).lower())


class BM25Index:
    """Small in-memory BM25 index; suitable for a frozen FEVER bank."""

    def __init__(self, bank: list[dict[str, Any]], k1: float = 1.2, b: float = 0.75):
        self.bank = bank
        self.k1 = k1
        self.b = b
        self.documents = [self._document(item) for item in bank]
        self.term_counts = [Counter(doc) for doc in self.documents]
        self.lengths = [len(doc) for doc in self.documents]
        self.avgdl = sum(self.lengths) / max(1, len(self.lengths))
        document_frequency = Counter()
        for doc in self.documents:
            document_frequency.update(set(doc))
        self.idf = {
            term: math.log(1.0 + (len(bank) - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in document_frequency.items()
        }

    @staticmethod
    def _document(item: dict[str, Any]) -> list[str]:
        evidence = " ".join(str(x.get("text", "")) for x in item.get("evidence_bundle", []))
        # Claim gets a small implicit emphasis by repeating its terms once in
        # the query construction, while evidence remains searchable context.
        return tokenize(f"{item.get('claim', '')} {evidence}")

    def search_with_scores(self, query: str, k: int = 6, label_bias: str | None = None) -> list[tuple[dict[str, Any], float]]:
        query_terms = tokenize(query)
        query_counts = Counter(query_terms)
        scored: list[tuple[float, str, dict[str, Any]]] = []
        for item, counts, length in zip(self.bank, self.term_counts, self.lengths):
            score = 0.0
            for term, query_frequency in query_counts.items():
                if term not in counts:
                    continue
                numerator = counts[term] * (self.k1 + 1.0)
                denominator = counts[term] + self.k1 * (1.0 - self.b + self.b * length / max(1.0, self.avgdl))
                score += self.idf.get(term, 0.0) * numerator / denominator
                if query_frequency > 1:
                    score *= 1.0 + 0.05 * min(query_frequency - 1, 3)
            if label_bias and item.get("gold_label") == label_bias:
                score += 0.05
            scored.append((score, str(item.get("memory_id", "")), item))
        scored.sort(key=lambda row: (row[0], row[1]), reverse=True)
        return [(item, float(score)) for score, _, item in scored[:k]]

    def search(self, query: str, k: int = 6, label_bias: str | None = None) -> list[dict[str, Any]]:
        return [item for item, _ in self.search_with_scores(query, k, label_bias)]


class GMemorySemanticIndex:
    """Frozen cosine index matching G-Memory's SentenceTransformer retrieval.

    G-Memory embeds the persisted task_main and ranks it against the current
    task query. For FEVER, task_main is the historical claim; its evidence and
    trajectory are injected after retrieval. Labels stay metadata and are not
    used as a retrieval filter: SUPPORTS/REFUTES are answer classes here, not
    success/failure outcomes.
    """

    def __init__(self, bank: list[dict[str, Any]], model_name: str = "sentence-transformers/all-MiniLM-L6-v2", threshold: float = 0.3, encoder: Any | None = None):
        if encoder is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:  # pragma: no cover - runtime dependency
                raise RuntimeError("G-Memory semantic retrieval requires sentence-transformers; install p2_probe_llm/requirements.txt") from exc
            encoder = SentenceTransformer(model_name)
        self.bank = bank
        self.threshold = threshold
        self.model_name = model_name
        self.model = encoder
        self.documents = [self._document(item) for item in bank]
        self.embeddings = self._encode(self.documents)

    @staticmethod
    def _document(item: dict[str, Any]) -> str:
        if item.get("task_main"):
            return str(item["task_main"])
        return str(item.get("claim", ""))

    def _encode(self, texts: list[str]) -> np.ndarray:
        values = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
        array = np.asarray(values, dtype=np.float32)
        if array.ndim == 1:
            array = array.reshape(1, -1)
        # G-Memory uses cosine similarity. Normalize here as well so a
        # custom/local encoder cannot silently change the metric.
        norms = np.linalg.norm(array, axis=1, keepdims=True)
        return array / np.where(norms == 0, 1.0, norms)

    def search_with_scores(self, query: str, k: int = 1, label_bias: str | None = None) -> list[tuple[dict[str, Any], float]]:
        if not self.bank or k <= 0:
            return []
        vector = self._encode([query])[0]
        scores = self.embeddings @ vector
        ranked = sorted(((float(score), str(item.get("memory_id", "")), item) for score, item in zip(scores, self.bank)), key=lambda row: (row[0], row[1]), reverse=True)
        return [(item, score) for score, _, item in ranked if score >= self.threshold][:k]

    def search(self, query: str, k: int = 1, label_bias: str | None = None) -> list[dict[str, Any]]:
        return [item for item, _ in self.search_with_scores(query, k, label_bias)]


def role_query(claim: dict[str, Any], agent_id: str) -> tuple[str, str | None]:
    claim_text = str(claim.get("claim", ""))
    # E1 uses one shared claim-level retrieval profile. Agent roles still
    # differ in reasoning, but role-conditioned retrieval would confound the
    # memory intervention with a retrieval intervention.
    return claim_text, None
