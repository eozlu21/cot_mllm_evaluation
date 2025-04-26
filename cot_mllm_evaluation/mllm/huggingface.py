from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from PIL import Image
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
)

from .base import BaseMLLM, FewShotExample


@dataclass(slots=True)
class HuggingFaceMLLM(BaseMLLM):
    """Vision–language model backed by Hugging Face `transformers`.

    Works with models such as **Qwen/Qwen2.5‑VL‑7B‑Instruct** that expose the
    *ImageTextToText* architecture.
    """

    model_name: str
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu"
    generate_kwargs: dict[str, Any] | None = None

    # ------------------------------------------------------------------
    def __post_init__(self) -> None:  # noqa: D401 – pydocstyle quirk
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
        ).to(self.device)
        # default generation kwargs unless user overrides
        self.generate_kwargs = self.generate_kwargs or {"max_new_tokens": 1048}

    # ------------------------------------------------------------------
    @torch.inference_mode()
    def prompt(
            self,
            image: PIL.Image,
            prompt: str,
            *,
            fewshot: Sequence[FewShotExample] | None = None,
            temperature: float = 0.2,
    ) -> str:  # noqa: D401
        # Build text prompt with optional few‑shot blocks.
        pil_image = image

        processed = self.processor(
            text=prompt,
            images=pil_image,
            return_tensors="pt",
        ).to(self.device)

        
        input_ids = processed["input_ids"]
        attention_mask = processed["attention_mask"]

        outputs = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **self.generate_kwargs,
        )
        return self.processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
