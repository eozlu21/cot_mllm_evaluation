from __future__ import annotations

from pathlib import Path


def write_jsonl(path: Path, rows: list[dict]) -> None:
    """Tiny helper to dump *rows* (a list of plain dicts) to *path* as JSONL."""
    import json, io

    with io.open(path, "w", encoding="utf‑8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


from pathlib import Path
from io import BytesIO
from typing import Union
import requests
from PIL import Image

def load_pil(img: Union[str, Path, Image.Image]) -> Image.Image:
    """Normalize various image inputs into a PIL.Image."""
    if isinstance(img, Image.Image):
        return img.convert("RGB")
    img_str = str(img)
    if img_str.startswith("http://") or img_str.startswith("https://"):
        resp = requests.get(img_str, timeout=10)
        resp.raise_for_status()
        return Image.open(BytesIO(resp.content)).convert("RGB")
    return Image.open(img_str).convert("RGB")
