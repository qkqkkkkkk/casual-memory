from __future__ import annotations

import hashlib
import json
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class MemoryItem:
    memory_id: str
    claim: str
    evidence_bundle: tuple[str, ...]
    gold_label: str
    team_verdict: str
    team_correct: bool
    rationale_digest: str
    token_count: int
    source_agent_ids: tuple[str, ...]
    created_at_episode_idx: int
    is_synthetic: bool = False
    parent_memory_ids: tuple[str, ...] = ()
    kind: str = "ordinary"

    def render(self) -> str:
        return (
            f"Claim: {self.claim}\nEvidence: {self.evidence_bundle[0]}\n"
            f"Historical verdict: {self.team_verdict}\nReason: {self.rationale_digest}"
        )


class FrozenMemoryStore:
    def __init__(self, items: list[MemoryItem]):
        self.items = tuple(items)
        self._md5 = hashlib.md5(
            json.dumps([asdict(x) for x in self.items], sort_keys=True).encode()
        ).hexdigest()

    @property
    def md5(self) -> str:
        return self._md5

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(json.dumps(asdict(x)) for x in self.items) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "FrozenMemoryStore":
        items = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                raw = json.loads(line)
                raw["evidence_bundle"] = tuple(raw["evidence_bundle"])
                raw["source_agent_ids"] = tuple(raw["source_agent_ids"])
                raw["parent_memory_ids"] = tuple(raw.get("parent_memory_ids", []))
                items.append(MemoryItem(**raw))
        return cls(items)

    def validate(self, expected_md5: str) -> None:
        if self.md5 != expected_md5:
            raise RuntimeError(f"Frozen memory hash mismatch: {self.md5} != {expected_md5}")


def normalize(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def lexical_similarity(a: str, b: str) -> float:
    aa, bb = normalize(a), normalize(b)
    return len(aa & bb) / max(1, len(aa | bb))


class RoleRetriever:
    """Simple frozen lexical retriever with role-specific query templates."""

    templates = {
        "A1": "{claim} {evidence}",
        "A2": "contradiction of {claim}",
        "A3": "{claim}",
    }

    def __init__(self, store: FrozenMemoryStore, top_k: int = 6):
        self.store, self.top_k = store, top_k

    def retrieve(self, claim: str, evidence: str, agent_id: str) -> list[MemoryItem]:
        query = self.templates[agent_id].format(claim=claim, evidence=evidence)
        ranked = sorted(
            self.store.items,
            key=lambda item: (lexical_similarity(query, item.claim), item.memory_id),
            reverse=True,
        )
        return ranked[: self.top_k]


def build_placebo(item: MemoryItem, store: FrozenMemoryStore, rng: random.Random) -> MemoryItem:
    """Select a dissimilar, token-matched frozen item as I0(m)."""
    candidates = [x for x in store.items if x.memory_id != item.memory_id]
    candidates.sort(key=lambda x: (lexical_similarity(item.claim, x.claim), x.memory_id))
    target = min(candidates[: max(10, len(candidates) // 4)], key=lambda x: abs(x.token_count - item.token_count))
    return MemoryItem(
        memory_id=f"placebo-{item.memory_id}", claim=target.claim,
        evidence_bundle=target.evidence_bundle, gold_label=target.gold_label,
        team_verdict=target.team_verdict, team_correct=target.team_correct,
        rationale_digest=target.rationale_digest, token_count=item.token_count,
        source_agent_ids=item.source_agent_ids, created_at_episode_idx=item.created_at_episode_idx,
        is_synthetic=True, parent_memory_ids=(item.memory_id,), kind="placebo",
    )


def make_synthetic_benchmark(n_claims: int, seed: int = 42) -> tuple[list[dict], FrozenMemoryStore]:
    rng = random.Random(seed)
    topics = ["Ada Lovelace", "Marie Curie", "Alan Turing", "Katherine Johnson", "Grace Hopper"]
    claims = []
    items = []
    for idx in range(n_claims):
        topic = topics[idx % len(topics)]
        label = "SUPPORTS" if idx % 2 == 0 else "REFUTES"
        claim = f"{topic} research fact number {idx} is historically documented."
        evidence = f"Archived record {idx} discusses {topic} and the relevant historical fact."
        claims.append({"claim_id": f"c{idx:04d}", "claim": claim, "evidence": evidence, "gold_label": label})
        # Every claim has one ordinary and one coordination-trap memory. Retrieval can see both.
        for kind in ("ordinary", "trap"):
            mid = f"m{idx:04d}-{kind}"
            wrong = "REFUTES" if label == "SUPPORTS" else "SUPPORTS"
            memory_label = label if kind == "ordinary" else wrong
            items.append(MemoryItem(
                memory_id=mid, claim=claim if kind == "ordinary" else claim.replace("historically", "not historically"),
                evidence_bundle=(evidence,), gold_label=memory_label, team_verdict=memory_label,
                team_correct=(kind == "ordinary"),
                rationale_digest=("Prior evidence supports independent verification." if kind == "ordinary" else "A confident precedent favors this verdict."),
                token_count=32, source_agent_ids=("A1", "A2", "A3"), created_at_episode_idx=idx, kind=kind,
            ))
    return claims, FrozenMemoryStore(items)


def load_fever_benchmark(path: Path, n_claims: int, seed: int = 42) -> tuple[list[dict], FrozenMemoryStore]:
    """Load the binary SUPPORTS/REFUTES slice of a FEVER jsonl split.

    Evidence text is optional in the public FEVER jsonl, so the simulator uses
    the claim itself as the compact evidence summary while preserving labels.
    """
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        if raw.get("label") in {"SUPPORTS", "REFUTES"}:
            rows.append(raw)
    rng = random.Random(seed)
    by_label = {label: [r for r in rows if r["label"] == label] for label in ("SUPPORTS", "REFUTES")}
    for bucket in by_label.values(): rng.shuffle(bucket)
    half = n_claims // 2
    selected = by_label["SUPPORTS"][:half] + by_label["REFUTES"][:half]
    if len(selected) < n_claims:
        remainder = [r for r in rows if r not in selected]
        rng.shuffle(remainder); selected.extend(remainder[: n_claims - len(selected)])
    claims = []
    items = []
    for idx, raw in enumerate(selected):
        claim = str(raw["claim"]); label = raw["label"]
        evidence = f"FEVER evidence summary for claim: {claim}"
        claims.append({"claim_id": f"fever-{raw.get('id', idx)}", "claim": claim, "evidence": evidence, "gold_label": label})
        wrong = _flip_label(label)
        for kind, memory_label in (("ordinary", label), ("trap", wrong)):
            items.append(MemoryItem(
                memory_id=f"m{idx:04d}-{kind}", claim=claim if kind == "ordinary" else f"Contradictory precedent: {claim}",
                evidence_bundle=(evidence,), gold_label=memory_label, team_verdict=memory_label,
                team_correct=(kind == "ordinary"),
                rationale_digest=("Prior evidence supports independent verification." if kind == "ordinary" else "A confident precedent favors this verdict."),
                token_count=max(16, len(claim.split()) + 12), source_agent_ids=("A1", "A2", "A3"), created_at_episode_idx=idx, kind=kind,
            ))
    return claims, FrozenMemoryStore(items)


def _flip_label(label: str) -> str:
    return "REFUTES" if label == "SUPPORTS" else "SUPPORTS"
