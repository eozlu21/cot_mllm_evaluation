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
        self.generate_kwargs = self.generate_kwargs or {"max_new_tokens": 64}

    # ------------------------------------------------------------------
    @torch.inference_mode()
    def prompt(
            self,
            image: Path,
            *,
            fewshot: Sequence[FewShotExample] | None = None,
            temperature: float = 0.2,
    ) -> str:
        prompt_parts: list[str] = [
            "You are an art critic. Write an uncanny literal description of the cartoon.\n"
        ]
        all_images: list[Image.Image] = []

        if fewshot:
            for ex in fewshot:
                prompt_parts.append("<image>\n")
                prompt_parts.append(f"{ex.text}\n")
                if isinstance(ex.image, (str, Path)):
                    all_images.append(Image.open(ex.image))
                else:
                    all_images.append(ex.image)  # assuming ex.image is already a PIL.Image.Image

        prompt_parts.append("<image>\nDescription:")
        prompt_text = "".join(prompt_parts)

        all_images.append(Image.open(image))  # the actual test image

        inputs = self.processor(
            text=prompt_text,
            images=all_images,
            return_tensors="pt",
        ).to(self.device)

        outputs = self.model.generate(
            **inputs,
            temperature=temperature,
            **self.generate_kwargs,
        )
        return self.processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()

