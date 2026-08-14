"""JSON output formatter for transcripts."""

from __future__ import annotations

import json
from dataclasses import asdict

from ownscribe.transcription.models import Segment, TranscriptResult, Word


def format_transcript_json(result: TranscriptResult) -> str:
    """Format a transcript result as JSON."""
    return json.dumps(asdict(result), indent=2, ensure_ascii=False)


def parse_transcript_json(text: str) -> TranscriptResult | None:
    """Read back a transcript written by format_transcript_json.

    Returns None if *text* is not a transcript document, so callers can fall
    back to treating the file as plain text.
    """
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(data, dict) or not isinstance(data.get("segments"), list):
        return None

    segments: list[Segment] = []
    for raw in data["segments"]:
        if not isinstance(raw, dict):
            continue
        words = [
            Word(
                text=w.get("word", w.get("text", "")),
                start=float(w.get("start", 0.0)),
                end=float(w.get("end", 0.0)),
                speaker=w.get("speaker"),
                score=float(w.get("score", 0.0)),
            )
            for w in raw.get("words", [])
            if isinstance(w, dict)
        ]
        segments.append(
            Segment(
                text=str(raw.get("text", "")),
                start=float(raw.get("start", 0.0)),
                end=float(raw.get("end", 0.0)),
                speaker=raw.get("speaker"),
                words=words,
            )
        )

    return TranscriptResult(
        segments=segments,
        language=str(data.get("language", "")),
        duration=float(data.get("duration", 0.0)),
    )
