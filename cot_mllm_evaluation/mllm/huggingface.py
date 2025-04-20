from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from transformers import (
    AutoModelForVision2Seq,
    AutoProcessor,
)

from .base import BaseMLLM, FewShotExample

@dataclass(slots=True)
class HuggingFaceMLLM(BaseMLLM):
    model_name: str
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu"
    generate_kwargs: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model = AutoModelForVision2Seq.from_pretrained(self.model_name, torch_dtype=torch.float16).to(self.device)
        self.generate_kwargs = self.generate_kwargs or dict(max_new_tokens=64)

    @torch.inference_mode()
    def prompt(
            self,
            image: Path,
            *,
            fewshot: Sequence[FewShotExample] | None = None,
            temperature: float = 0.2,
    ) -> str:
        # Prompt structure may be different for Qwen2.5 — check tokenizer requirements
        prompt_parts = [
            "You are an art critic. Write an uncanny literal description of the cartoon.\n"
        ]
        if fewshot:
            for ex in fewshot:
                prompt_parts.append(f"<img>\n{ex.text}\n")

        prompt_parts.append("<img>\nDescription:")
        prompt = "".join(prompt_parts)

        inputs = self.processor(text=prompt, images=image.open("rb"), return_tensors="pt").to(self.device)
        outputs = self.model.generate(**inputs, temperature=temperature, **self.generate_kwargs)
        return self.processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
