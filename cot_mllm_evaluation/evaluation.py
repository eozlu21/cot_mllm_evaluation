from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Iterable

import datasets

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
            answer_field: str = "uncanny_description",
    ) -> None:
        self.dataset = datasets.load_dataset(dataset_name, split="train")  # type: ignore[arg‑type]
        self.mllm = mllm
        self.verifier = verifier
        self.fewshot = list(fewshot or [])
        self.answer_field = answer_field
        self.stats = Counter()

    # --------------------------------------------------
    def run(self) -> None:
        for row in self.dataset:
            image_path: Path = Path(row["image"])
            gold: str = str(row[self.answer_field]).strip()
            guess: str = self.mllm.prompt(image_path, fewshot=self.fewshot)
            correct: bool = self.verifier.verify(gold, guess)
            self.stats["total"] += 1
            if correct:
                self.stats["correct"] += 1

    # --------------------------------------------------
    @property
    def accuracy(self) -> float:
        return self.stats["correct"] / max(1, self.stats["total"])
