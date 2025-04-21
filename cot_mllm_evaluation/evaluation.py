from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Iterable

import datasets
from PIL import Image  # Add this import if not already present
import tempfile

from .mllm.base import BaseMLLM, FewShotExample
from .verifier.base import BaseVerifier


class Evaluator:
    """Runs the full pipeline and keeps a running tally."""

    def __init__(
            self,
            dataset_name: str,
            *,
            mllm: BaseMLLM,
            verifier: BaseVerifier,
            fewshot: Iterable[FewShotExample] | None = None,
            answer_field: str = "image_uncanny_description",
            num_samples: int | None = None,
    ) -> None:
        if num_samples is None:
            self.dataset = datasets.load_dataset(dataset_name,name= "explanation" ,split="train")
        else:
            self.dataset = datasets.load_dataset(dataset_name,name= "explanation" ,split="train").select(range(num_samples)) # type: ignore[arg‑type]
        self.mllm = mllm
        self.verifier = verifier
        self.fewshot = list(fewshot or [])
        self.answer_field = answer_field
        self.stats = Counter()

    # --------------------------------------------------
    def run(self) -> None:
        for row in self.dataset:
            #print("Row:", row)
            image_raw = row.get("image")
            if not image_raw:
                print(f"Skipping row with missing image: {row}")
                continue

            # Handle JpegImageFile objects
            if isinstance(image_raw, Image.Image):
                with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
                    image_raw.save(temp_file.name)
                    image_path: Path = Path(temp_file.name)
            else:
                image_path: Path = Path(image_raw)

            # Check if the answer field exists in the row
            if self.answer_field not in row:
                print(f"Skipping row with missing answer field '{self.answer_field}': {row}")
                continue

            gold: str = str(row[self.answer_field]).strip()
            
            guess: str = self.mllm.prompt(image_path, fewshot=self.fewshot)
            correct: bool = self.verifier.verify(gold, guess)
            print(f"Image: {image_path}\nGold: {gold}\nGuess: {guess}\nCorrect: {correct}")
            self.stats["total"] += 1
            if correct:
                self.stats["correct"] += 1

    # --------------------------------------------------
    @property
    def accuracy(self) -> float:
        return self.stats["correct"] / max(1, self.stats["total"])
