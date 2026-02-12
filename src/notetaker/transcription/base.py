"""Abstract base class for transcribers."""

from __future__ import annotations

import abc
from pathlib import Path

from notetaker.transcription.models import TranscriptResult


class Transcriber(abc.ABC):
    """Base class for transcription backends."""

    @abc.abstractmethod
    def transcribe(self, audio_path: Path) -> TranscriptResult:
        """Transcribe an audio file and return structured results."""
