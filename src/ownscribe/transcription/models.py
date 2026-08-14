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
    def speaker_text(self) -> str:
        """Transcript text with a speaker label at every speaker change.

        Consecutive segments from the same speaker are joined into one line, so
        the summarizer can tell who said what. Falls back to full_text when
        diarization did not run and no segment carries a speaker.
        """
        if not self.has_speakers:
            return self.full_text

        lines: list[str] = []
        current_speaker: str | None = None
        parts: list[str] = []

        for seg in self.segments:
            text = seg.text.strip()
            if not text:
                continue
            if parts and seg.speaker != current_speaker:
                lines.append(f"{current_speaker or 'Unknown'}: {' '.join(parts)}")
                parts = []
            current_speaker = seg.speaker
            parts.append(text)

        if parts:
            lines.append(f"{current_speaker or 'Unknown'}: {' '.join(parts)}")
        return "\n".join(lines)

    @property
    def has_speakers(self) -> bool:
        return any(seg.speaker is not None for seg in self.segments)
