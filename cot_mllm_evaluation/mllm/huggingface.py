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
from transformers import Qwen2_5_VLForConditionalGeneration

from typing import List, Tuple
from cot_mllm_evaluation.mllm.base import BaseMLLM, FewShotExample

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
    """
    def __post_init__(self) -> None:  # noqa: D401 – pydocstyle quirk
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
        ).to(self.device)
        # default generation kwargs unless user overrides
        self.generate_kwargs = self.generate_kwargs or {"max_new_tokens": 1048}
    """
        
    def __post_init__(self) -> None:

        if self.model_name == "Fancy-MLLM/R1-Onevision-7B":
            self.processor = AutoProcessor.from_pretrained(self.model_name, trust_remote_code=True)
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16
            ).to("cuda").eval()
        else:
            self.processor = AutoProcessor.from_pretrained(self.model_name)
            self.model = AutoModelForImageTextToText.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
            ).to(self.device)
        self.image_token = getattr(self.processor.tokenizer, "image_token", "<|image|>")
        self.generate_kwargs = self.generate_kwargs or {"max_new_tokens": 1048}


    # ------------------------------------------------------------------

    @staticmethod
    def process_vision_info(messages: List[dict]) -> Tuple[List[Image.Image], List]:
        """Extract image and video inputs from messages."""
        image_inputs = []
        video_inputs = []

        for msg in messages:
            for content in msg.get("content", []):
                if content.get("type") == "image":
                    image_inputs.append(content["image"])
                elif content.get("type") == "video":
                    video_inputs.append(content["video"])
        return image_inputs, video_inputs


    @torch.inference_mode()
    def prompt(
        self,
        image: PIL.Image.Image,
        prompt: str,
        *,
        fewshot: Sequence[FewShotExample] | None = None,
        temperature: float = 0.2,
    ) -> str:
        # Build Qwen-style messages
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        # Format chat template
        text_prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Extract image inputs properly
        image_inputs, video_inputs = self.process_vision_info(messages)

        # Tokenize
        inputs = self.processor(
            text=[text_prompt],
            images=image_inputs,
            **({"videos": video_inputs} if video_inputs else {}),
            padding=True,
            return_tensors="pt",
        ).to(self.device)


        # Generate
        generated_ids = self.model.generate(**inputs, **self.generate_kwargs)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        return output[0].strip()
