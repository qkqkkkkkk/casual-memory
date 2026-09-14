"""FEVER/P2 backend for the generic counterfactual audit contracts.

This module deliberately reuses the real ``p2_probe_llm`` prompts, cache, and
two-round team implementation.  The frozen prefix is the selected claim,
evidence bundle, semantic retrieval, and per-agent candidate sets.  Branches
start immediately before round-1 prompt injection.
"""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Any, Mapping, Optional

from p2_probe_llm.mas import _render_memory, run_episode

from .types import (
    AuditCheckpoint,
    BranchOutcome,
    BranchRequest,
    MemoryCandidate,
    MemoryUseEvent,
    RecipientContext,
    RetrievalMetadata,
)


AGENTS = ("A1", "A2", "A3")


def memory_candidate(item: Mapping[str, Any]) -> MemoryCandidate:
    return MemoryCandidate(
        memory_id=str(item["memory_id"]),
        memory_type="trajectory",
        content=_render_memory(dict(item)),
        metadata={
            "historical_success": bool(item.get("historical_success", True)),
            "retrieval_score": item.get("retrieval_score"),
            "retrieval_rank": item.get("retrieval_rank"),
            "source_example_id": item.get("source_example_id"),
            "source_task_type": "fever",
            "source_model_id": item.get("source_model_id"),
            "memory_schema_version": item.get(
                "memory_schema_version", "gmemory-fever-v2"
            ),
        },
    )


def build_fever_event(
    claim: Mapping[str, Any],
    candidates: Mapping[str, list[dict[str, Any]]],
    *,
    receiver: str,
    target_id: str,
    event_id: str,
) -> MemoryUseEvent:
    registry: dict[str, MemoryCandidate] = {}
    for values in candidates.values():
        for item in values:
            registry.setdefault(str(item["memory_id"]), memory_candidate(item))
    if target_id not in registry:
        raise KeyError(f"target memory {target_id!r} is absent")
    target = registry[target_id]
    contexts = tuple(
        RecipientContext(
            agent_id,
            tuple(str(item["memory_id"]) for item in candidates[agent_id]),
        )
        for agent_id in AGENTS
    )
    evidence = "\n".join(
        str(entry.get("text", "")) for entry in claim.get("evidence_bundle", [])
    )
    score = target.metadata.get("retrieval_score")
    rank = target.metadata.get("retrieval_rank")
    return MemoryUseEvent(
        event_id=event_id,
        query=str(claim.get("claim", "")),
        task_state=evidence,
        receiver_agent_id=receiver,
        receiver_role=receiver,
        memory=target,
        candidate_set=tuple(registry.values()),
        recipient_contexts=contexts,
        retrieval=RetrievalMetadata(
            candidate_id=target_id,
            source="gmemory_semantic_claim",
            rank=int(rank) if rank is not None else None,
            similarity=float(score) if score is not None else None,
        ),
        task_metadata={
            "task_type": "fever",
            "claim_id": str(claim.get("id", "")),
            "evidence_count": len(claim.get("evidence_bundle", [])),
            "agent_count": len(AGENTS),
            "round_count": 2,
        },
        reliability_prior=float(bool(target.metadata.get("historical_success"))),
    )


def build_fever_checkpoint(event: MemoryUseEvent, claim_id: str) -> AuditCheckpoint:
    return AuditCheckpoint(
        checkpoint_id=f"{event.event_id}-pre-round1",
        event_id=event.event_id,
        upstream_state={
            "claim_id": claim_id,
            "recipient_candidate_ids": {
                context.agent_id: context.candidate_ids
                for context in event.recipient_contexts
            },
        },
        sampling_config={
            "temperature": 0.7,
            "top_p": 0.8,
            "response_format": "json_object",
        },
    )


def placebo_candidate(item: Mapping[str, Any]) -> MemoryCandidate:
    candidate = memory_candidate(item)
    return MemoryCandidate(
        memory_id=candidate.memory_id,
        memory_type=candidate.memory_type,
        content=candidate.content,
        metadata={**candidate.metadata, "irrelevant": True},
    )


class FeverBranchRunner:
    """Run/resume one FEVER event branch and append the raw episode to JSONL."""

    def __init__(
        self,
        *,
        client: Any,
        claim: dict[str, Any],
        candidates: Mapping[str, list[dict[str, Any]]],
        placebo_item: Optional[dict[str, Any]],
        log_path: Path,
        design_hash: str,
        run_hash: str,
        existing_rows: Optional[dict[tuple[str, int, str], dict[str, Any]]] = None,
    ):
        self.client = client
        self.claim = claim
        self.candidates = {key: list(value) for key, value in candidates.items()}
        self.log_path = log_path
        self.design_hash = design_hash
        self.run_hash = run_hash
        self.existing_rows = existing_rows if existing_rows is not None else {}
        self.raw_by_id: dict[str, dict[str, Any]] = {
            str(item["memory_id"]): item
            for values in self.candidates.values()
            for item in values
        }
        if placebo_item is not None:
            self.raw_by_id[str(placebo_item["memory_id"])] = placebo_item

    def run(self, request: BranchRequest) -> BranchOutcome:
        key = (request.event.event_id, request.repeat_index, request.arm.value)
        existing = self.existing_rows.get(key)
        if existing is not None:
            self._validate_existing(existing, request)
            return self._outcome(existing, request.event.receiver_agent_id)

        actual: dict[str, list[dict[str, Any]]] = {}
        for context in request.recipient_contexts:
            try:
                actual[context.agent_id] = [
                    self.raw_by_id[memory_id] for memory_id in context.candidate_ids
                ]
            except KeyError as exc:
                raise KeyError(
                    f"raw FEVER memory {exc.args[0]!r} is unavailable"
                ) from exc
        missing_agents = set(AGENTS) - set(actual)
        if missing_agents:
            raise ValueError(f"branch is missing agents: {sorted(missing_agents)}")

        event = request.event
        audit_unit = (
            str(self.claim["id"]),
            event.receiver_agent_id,
            event.memory.memory_id,
        )
        cache_hits_before = int(getattr(self.client, "cache_hits", 0))
        calls_before = int(getattr(self.client, "calls", 0))
        episode = run_episode(
            self.client,
            self.claim,
            actual,
            request.seed,
            request.arm.value,
            audit_unit,
            placebo_lookup=None,
        )
        row = {
            "runner_schema": "causal-memory-control-fever-v1",
            "design_hash": self.design_hash,
            "run_hash": self.run_hash,
            "event_id": event.event_id,
            "checkpoint_id": request.checkpoint.checkpoint_id,
            "branch": request.arm.value,
            "repeat_index": request.repeat_index,
            "sample_seed": request.seed,
            "receiver_agent_id": event.receiver_agent_id,
            "target_memory_id": event.memory.memory_id,
            "memory_observation_count": event.observation_count,
            "recipient_candidate_ids": {
                context.agent_id: list(context.candidate_ids)
                for context in request.recipient_contexts
            },
            **asdict(episode),
            "branch_cache_hits": int(getattr(self.client, "cache_hits", 0))
            - cache_hits_before,
            "branch_llm_calls": int(getattr(self.client, "calls", 0)) - calls_before,
        }
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
        self.existing_rows[key] = row
        return self._outcome(row, event.receiver_agent_id)

    @staticmethod
    def _outcome(row: Mapping[str, Any], receiver: str) -> BranchOutcome:
        receiver_output = row["round1"][receiver]
        return BranchOutcome(
            response=str(receiver_output["verdict"]),
            behavior_payload=str(receiver_output["verdict"]),
            team_reward=float(bool(row["team_correct"])),
            local_metric=float(bool(receiver_output["correct"])),
            metadata={"raw_run": dict(row)},
        )

    def _validate_existing(
        self, row: Mapping[str, Any], request: BranchRequest
    ) -> None:
        expected_contexts = {
            context.agent_id: list(context.candidate_ids)
            for context in request.recipient_contexts
        }
        if row.get("design_hash") != self.design_hash:
            raise ValueError("existing row has a different design_hash")
        if row.get("run_hash") != self.run_hash:
            raise ValueError("existing row has a different run_hash")
        if row.get("sample_seed") != request.seed:
            raise ValueError("existing row has a different sample seed")
        if row.get("recipient_candidate_ids") != expected_contexts:
            raise ValueError("existing row has different recipient candidate sets")


def load_existing_rows(path: Path) -> dict[tuple[str, int, str], dict[str, Any]]:
    rows: dict[tuple[str, int, str], dict[str, Any]] = {}
    if not path.exists():
        return rows
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        key = (str(row["event_id"]), int(row["repeat_index"]), str(row["branch"]))
        if key in rows:
            raise ValueError(f"duplicate audit branch at line {line_number}: {key}")
        rows[key] = row
    return rows
