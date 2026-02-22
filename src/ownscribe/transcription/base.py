"""Abstract base class for transcribers."""

from __future__ import annotations

import abc
from pathlib import Path

from ownscribe.transcription.models import TranscriptResult


class Transcriber(abc.ABC):
    """Base class for transcription backends."""

    def prepare_models(self, language: str | None = None) -> None:
        """Optional hook to prefetch/load models before transcription."""
        _ = language

    @abc.abstractmethod
    def transcribe(self, audio_path: Path) -> TranscriptResult:
        """Transcribe an audio file and return structured results."""
