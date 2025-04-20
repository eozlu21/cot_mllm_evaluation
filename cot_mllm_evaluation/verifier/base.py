from __future__ import annotations

from abc import ABC, abstractmethod


class BaseVerifier(ABC):
    """Checks whether *prediction* matches *reference* using an LLM‑backed judge."""

    @abstractmethod
    def verify(self, reference: str, prediction: str) -> bool:  # noqa: D401
        """Return *True* if prediction ≈ reference, *False* otherwise."""
