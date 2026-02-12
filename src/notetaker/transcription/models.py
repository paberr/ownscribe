"""Data models for transcription results."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Word:
    text: str
    start: float
    end: float
    speaker: str | None = None
    score: float = 0.0


@dataclass
class Segment:
    text: str
    start: float
    end: float
    speaker: str | None = None
    words: list[Word] = field(default_factory=list)


@dataclass
class TranscriptResult:
    segments: list[Segment]
    language: str = ""
    duration: float = 0.0

    @property
    def full_text(self) -> str:
        return " ".join(seg.text.strip() for seg in self.segments)

    @property
    def has_speakers(self) -> bool:
        return any(seg.speaker is not None for seg in self.segments)
