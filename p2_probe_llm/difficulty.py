from __future__ import annotations

import hashlib
from typing import Any

from .retrieval import tokenize


def sim(a: str, b: str) -> float:
    aa, bb = set(tokenize(a)), set(tokenize(b))
    return len(aa & bb) / max(1, len(aa | bb))


def make_variant(claim: dict[str, Any], bank: list[dict[str, Any]], gold_recall: float, seed: int, max_sentences: int = 5) -> dict[str, Any]:
    digest = hashlib.sha256(f"{seed}|{claim['id']}".encode()).hexdigest()
    include_gold = int(digest[:16], 16) / (16 ** 16) < gold_recall
    bundle = list(claim.get("evidence_bundle", []))[:2] if include_gold else []
    ranked = sorted((item for item in bank if item.get("claim") != claim.get("claim")), key=lambda item: (sim(claim["claim"], item.get("claim", "")), str(item.get("memory_id"))), reverse=True)
    seen = {entry.get("text") for entry in bundle}
    for item in ranked:
        for entry in item.get("evidence_bundle", []):
            text = entry.get("text", "")
            if text and text not in seen:
                bundle.append({"title": f"Distractor from {item['memory_id']}", "line_id": None, "text": text, "is_gold": False})
                seen.add(text)
                break
        if len(bundle) >= max_sentences: break
    result = dict(claim); result["evidence_bundle"] = bundle[:max_sentences]
    result["evidence_policy"] = {"gold_recall": gold_recall, "gold_included": include_gold, "max_sentences": max_sentences, "seed": seed}
    return result
