#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .evidence import enrich_split


def main() -> None:
    p = argparse.ArgumentParser(description="Resolve FEVER evidence IDs to Wikipedia sentences")
    p.add_argument("--input", type=Path, required=True, help="FEVER JSONL split")
    p.add_argument("--wiki", type=Path, required=True, help="FEVER wiki-pages JSONL or converted dump")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--max-sentences", type=int, default=5)
    args = p.parse_args()
    print(json.dumps(enrich_split(args.input, args.wiki, args.output, args.max_sentences), indent=2))


if __name__ == "__main__": main()
