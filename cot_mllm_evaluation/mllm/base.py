from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Sequence, Protocol
from dataclasses import dataclass


@dataclass(slots=True)
class FewShotExample():
    """A structural type for few‑shot examples."""

    image: Path
    text: str


class BaseMLLM(ABC):
    """Common interface for any multimodal LLM we might want to swap in/out."""

    @abstractmethod
    def prompt(
            self,
            image: Path,
            *,
            fewshot: Sequence[FewShotExample] | None = None,
            temperature: float = 0.2,
    ) -> str:
        """Return the model’s description for *image*.

        *fewshot* – optional (image, text) pairs to help the model catch on to the
        task.  Implementations decide how these are woven into the prompt.
        """

