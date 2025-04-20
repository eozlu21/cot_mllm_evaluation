from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence, Union

import torch
from PIL import Image
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
)

from .base import BaseMLLM, FewShotExample
from ..utils import load_pil


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
        # 1) Build a “chat” structure with image/text messages
        messages = []
        images: list[Image.Image] = []

        if fewshot:
            for ex in fewshot:
                # one user turn showing the example image + text
                messages.append({
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": ex.text}
                    ],
                })
                images.append(load_pil(ex.image))

        # final user turn: your actual prompt
        messages.append({
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "You are an art critic. Write an uncanny literal description of the cartoon."}
            ],
        })
        images.append(load_pil(image))

        # 2) Let the HF processor build the input_ids + align images
        chat_text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.processor(
            text=[chat_text],
            images=images,
            return_tensors="pt",
        ).to(self.device)

        # 3) Generate and decode
        outputs = self.model.generate(
            **inputs,
            temperature=temperature,
            **self.generate_kwargs,
        )
        return self.processor.batch_decode(
            outputs,
            skip_special_tokens=True
        )[0].strip()

