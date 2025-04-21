from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .base import BaseVerifier


@dataclass(slots=True)
class LLMVerifier(BaseVerifier):
    model_name: str = "gpt2"
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu"

    def __post_init__(self) -> None:  # noqa: D401
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name).to(self.device)

    # --------------------------------------------------
    # public API
    # --------------------------------------------------
    @torch.inference_mode()
    def verify(self, reference: str, prediction: str) -> bool:  # noqa: D401
        prompt = (
            "You are a strict grader.  Answer only with ‘yes’ or ‘no’.\n"
            f"Reference: {reference}\n"
            f"Prediction: {prediction}\n"
            "Are these two descriptions semantically equivalent?"
        )
        encoded = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        generated = self.model.generate(**encoded, max_new_tokens=1)
        print(f"Prompt provided to verifier: {prompt}")
        answer = self.tokenizer.decode(generated[0], skip_special_tokens=False).lower()
        print(f"Generated answer by verifier: {answer}")
        return "yes" in answer