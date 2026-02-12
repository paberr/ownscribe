"""JSON output formatter for transcripts."""

from __future__ import annotations

import json
from dataclasses import asdict

from notetaker.transcription.models import TranscriptResult


def format_transcript_json(result: TranscriptResult) -> str:
    """Format a transcript result as JSON."""
    return json.dumps(asdict(result), indent=2, ensure_ascii=False)
