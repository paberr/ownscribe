"""Abstract base class for summarizers."""

from __future__ import annotations

import abc


class Summarizer(abc.ABC):
    """Base class for summarization backends."""

    @abc.abstractmethod
    def summarize(self, transcript_text: str) -> str:
        """Summarize a transcript and return the summary text."""

    @abc.abstractmethod
    def is_available(self) -> bool:
        """Check if the summarization backend is reachable."""
