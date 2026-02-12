"""Abstract base class for audio recorders."""

from __future__ import annotations

import abc
from pathlib import Path


class AudioRecorder(abc.ABC):
    """Base class for audio recording backends."""

    @abc.abstractmethod
    def start(self, output_path: Path) -> None:
        """Start recording audio to the given path."""

    @abc.abstractmethod
    def stop(self) -> None:
        """Stop recording."""

    @abc.abstractmethod
    def is_available(self) -> bool:
        """Check if this backend is available on the current system."""
