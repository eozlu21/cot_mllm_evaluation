from __future__ import annotations

from pathlib import Path


def write_jsonl(path: Path, rows: list[dict]) -> None:
    """Tiny helper to dump *rows* (a list of plain dicts) to *path* as JSONL."""
    import json, io

    with io.open(path, "w", encoding="utf‑8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")