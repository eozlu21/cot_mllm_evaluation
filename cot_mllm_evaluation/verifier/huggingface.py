from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .base import BaseVerifier


@dataclass(slots=True)
class LLMVerifier(BaseVerifier):
    model_name: str = "gpt2"
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu"

    def __post_init__(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name).to(self.device)

    @torch.inference_mode()
    def verify(self, reference: str, prediction: str) -> bool:
        # Hard-coded few-shot examples
        prompt = (
            "You are a strict grader. Answer only with yes or no (lowercase), nothing else. You will answer this question: Are the reference and prediction semantically equivalent?\n"
            "Example 1:\n"
            "Reference: A doctor treats patients in a hospital.\n"
            "Prediction: A physician cares for individuals in a healthcare facility.\n"
            "Assistant: yes\n\n"
            "Example 2:\n"
            "Reference: The sky is green.\n"
            "Prediction: The grass is blue.\n"
            "Assistant: no\n\n"
            f"Reference: {reference}\n"
            f"Prediction: {prediction}\n"
            "Assistant:"
        )

        encoded = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        # Greedy, deterministic decoding for a single token
        output_ids = self.model.generate(
            **encoded,
            max_new_tokens=1,
            do_sample=False,
            temperature=0.0,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        # Extract only the newly generated token(s)
        new_tokens = output_ids[0, encoded.input_ids.shape[-1]:]
        answer = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip().lower()

        # Debug prints (optional)
        print(f"Prompt provided to verifier: {prompt}\n")
        print(f"Generated raw tokens: {new_tokens}\n")
        print(f"Decoded answer: {answer}\n")

        return answer == "yes"
