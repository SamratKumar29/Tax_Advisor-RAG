"""
Structured per-query log: one JSON line per answered question with per-stage
latency, model, token usage, retrieved chunk ids, abstention and citation result.

Written to QUERY_LOG_PATH (default logs/queries.jsonl, git-ignored). The log
contains users' questions; set QUERY_LOG_PATH="" to disable (e.g. shared demos).
Logging never raises into the answer path.
"""

import json
from datetime import datetime, timezone

from src import config


def log_query(query: str, result: dict, timings: dict, chunk_ids: list[str] | None = None) -> None:
    if not config.QUERY_LOG_PATH:
        return
    check = result.get("citation_check") or {}
    record = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "query": query,
        "search_query": result.get("search_query", query),
        "abstained": result.get("abstained", False),
        "confidence": result.get("confidence"),
        "model": result.get("model_used", ""),
        "usage": result.get("usage"),
        "timings_ms": timings,
        "chunk_ids": chunk_ids or [],
        "citation_ok": check.get("ok"),
        "citation_unsupported": len(check.get("unsupported", [])),
    }
    try:
        path = config.resolve_path(config.QUERY_LOG_PATH)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    except OSError:
        pass
