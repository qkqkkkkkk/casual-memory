from __future__ import annotations

import gzip
import json
import re
from pathlib import Path
from typing import Any


def norm_title(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()


def evidence_refs(raw: dict[str, Any]) -> list[tuple[str, int | None]]:
    refs: list[tuple[str, int | None]] = []
    for group in raw.get("evidence", []) or []:
        if not isinstance(group, list):
            continue
        for item in group:
            if not isinstance(item, list):
                continue
            title = next((item[i] for i in (2, 1) if i < len(item) and isinstance(item[i], str) and item[i].strip()), None)
            line_id = next((item[i] for i in (3, 2, 1) if i < len(item) and isinstance(item[i], int)), None)
            if title:
                refs.append((title, line_id))
    return list(dict.fromkeys(refs))


class WikiIndex:
    """Scan FEVER wiki shards while retaining only pages referenced by a split."""

    def __init__(self, path: Path, wanted_titles: set[str]):
        self.pages: dict[str, dict[int, str]] = {}
        self._load(path, {norm_title(x) for x in wanted_titles})

    def _load(self, path: Path, wanted: set[str]) -> None:
        sources = sorted(path.glob("*.jsonl*")) if path.is_dir() else [path]
        if not sources:
            raise FileNotFoundError(f"No JSONL wiki shards found under {path}")
        for source in sources:
            opener = gzip.open if source.suffix == ".gz" else open
            with opener(source, "rt", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    raw = json.loads(line)
                    title = raw.get("id") or raw.get("title") or raw.get("page")
                    key = norm_title(title or "")
                    if key and key in wanted:
                        self.pages[key] = self._parse_lines(raw)

    @staticmethod
    def _parse_lines(raw: dict[str, Any]) -> dict[int, str]:
        value = raw.get("lines")
        parsed: dict[int, str] = {}
        if isinstance(value, str):
            iterable = value.splitlines()
        elif isinstance(value, list):
            iterable = value
        else:
            iterable = str(raw.get("text") or "").splitlines()
        for fallback, item in enumerate(iterable):
            if isinstance(item, dict):
                line_id = int(item.get("line_id", item.get("id", fallback)))
                text = str(item.get("text") or item.get("sentence") or "")
            else:
                parts = str(item).split("\t")
                line_id = int(parts[0]) if parts and parts[0].isdigit() else fallback
                text = parts[1] if len(parts) > 1 else str(item)
            parsed[line_id] = text.strip()
        return parsed

    def get(self, title: str, line_id: int | None) -> str | None:
        lines = self.pages.get(norm_title(title), {})
        if line_id is not None and line_id in lines:
            return lines[line_id]
        return next((text for text in lines.values() if text), None)


def enrich_split(input_path: Path, wiki_path: Path, output_path: Path, max_sentences: int = 5) -> dict[str, int]:
    rows = [json.loads(line) for line in input_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    wanted = {title for raw in rows for title, _ in evidence_refs(raw)}
    index = WikiIndex(wiki_path, wanted)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    total = found = binary = binary_with_evidence = 0
    try:
        dst = output_path.open("x", encoding="utf-8")
    except FileExistsError as exc:
        raise RuntimeError(f"Refusing to overwrite enriched split: {output_path}") from exc
    with dst:
        for raw in rows:
            total += 1
            bundle = []
            for title, line_id in evidence_refs(raw):
                sentence = index.get(title, line_id)
                if sentence:
                    bundle.append({"title": title, "line_id": line_id, "text": sentence})
            label = raw.get("label")
            found += int(bool(bundle))
            if label in {"SUPPORTS", "REFUTES"}:
                binary += 1
                binary_with_evidence += int(bool(bundle))
            enriched = dict(raw)
            enriched["evidence_bundle"] = bundle[:max_sentences]
            enriched["evidence_found"] = bool(bundle)
            dst.write(json.dumps(enriched, ensure_ascii=False) + "\n")
    return {"examples": total, "with_evidence": found, "binary": binary, "binary_with_evidence": binary_with_evidence, "referenced_pages": len(wanted), "resolved_pages": len(index.pages)}
