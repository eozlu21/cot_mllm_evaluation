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

    def __post_init__(self) -> None:
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
        ).to(self.device)
        self.generate_kwargs = self.generate_kwargs or {"max_new_tokens": 256}

    @torch.inference_mode()
    def prompt(
            self,
            image: Path,
            *,
            fewshot: Sequence[FewShotExample] | None = None,
            temperature: float = 0.2,
    ) -> str:
        messages: list[dict[str, Union[str, list[dict[str, str]]]]] = []
        images: list[Image.Image] = []

        if fewshot:
            for ex in fewshot:
                messages.append({
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": ex.text}
                    ],
                })
                images.append(load_pil(ex.image))

        messages.append({
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "You are a comedy expert. Write the uncanny description of the cartoon. You are essentially finding out what is unusual in this cartoon that could be used to come up with a funny caption. Write one sentence only."}
            ],
        })
        images.append(load_pil(image))

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

        outputs = self.model.generate(
            **inputs,
            temperature=temperature,
            **self.generate_kwargs,
        )

        # Slice off the prompt tokens to keep only generated response
        input_length = inputs.input_ids.shape[-1]
        gen_ids = outputs[0, input_length:]
        response = self.processor.batch_decode(
            [gen_ids],
            skip_special_tokens=True,
        )[0].strip()
        return response
