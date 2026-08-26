from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any


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

    def search(self, query: str, k: int = 6, label_bias: str | None = None) -> list[dict[str, Any]]:
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
        return [item for _, _, item in scored[:k]]


def role_query(claim: dict[str, Any], agent_id: str) -> tuple[str, str | None]:
    claim_text = str(claim.get("claim", ""))
    evidence_text = " ".join(str(x.get("text", "")) for x in claim.get("evidence_bundle", []))
    if agent_id == "A1":
        return f"{claim_text} {evidence_text}", None
    if agent_id == "A2":
        return f"contradiction counterexample evidence gap {claim_text}", "REFUTES"
    return f"historical precedent prior claim {claim_text}", None
