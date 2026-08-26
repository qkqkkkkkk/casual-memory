from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time
import urllib.request
from pathlib import Path
from typing import Any


class CachedChat:
    def __init__(self, endpoint: str, model: str, cache_path: Path, temperature: float = 0.7, timeout: int = 180, api_key: str | None = None):
        self.endpoint = endpoint.rstrip("/") + "/chat/completions"
        self.model, self.temperature, self.timeout = model, temperature, timeout
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(cache_path), timeout=60)
        self.db.execute("CREATE TABLE IF NOT EXISTS responses (key TEXT PRIMARY KEY, response TEXT NOT NULL)")
        self.db.commit()
        self.cache_hits = 0; self.calls = 0

    def ask(self, messages: list[dict[str, str]], repeat_idx: int) -> tuple[dict[str, Any], bool]:
        payload = {"model": self.model, "messages": messages, "temperature": self.temperature, "top_p": 0.8, "seed": repeat_idx, "max_tokens": 320, "response_format": {"type": "json_object"}}
        key = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        cached = self.db.execute("SELECT response FROM responses WHERE key=?", (key,)).fetchone()
        if cached:
            self.cache_hits += 1; return json.loads(cached[0]), True
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(self.endpoint, data=json.dumps(payload).encode(), headers=headers, method="POST")
        for attempt in range(4):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    raw = json.loads(response.read().decode())
                content = raw["choices"][0]["message"]["content"]
                if isinstance(content, str):
                    cleaned = content.strip()
                    if cleaned.startswith("```"):
                        cleaned = cleaned.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
                    parsed = json.loads(cleaned)
                else:
                    parsed = content
                break
            except Exception:
                if attempt == 3: raise
                time.sleep(2 ** attempt)
        self.db.execute("INSERT OR REPLACE INTO responses VALUES (?,?)", (key, json.dumps(parsed, ensure_ascii=False))); self.db.commit()
        self.calls += 1
        return parsed, False
