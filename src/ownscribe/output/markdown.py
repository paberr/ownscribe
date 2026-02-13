"""Markdown output formatter for transcripts."""

from __future__ import annotations

from ownscribe.transcription.models import TranscriptResult


def _format_time(seconds: float) -> str:
    """Format seconds as HH:MM:SS or MM:SS."""
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def format_transcript(result: TranscriptResult) -> str:
    """Format a transcript result as markdown."""
    lines = ["# Transcript\n"]

    if result.language:
        lines.append(f"**Language:** {result.language}  ")
    if result.duration > 0:
        lines.append(f"**Duration:** {_format_time(result.duration)}  ")
    lines.append("")

    current_speaker = None
    for seg in result.segments:
        timestamp = f"[{_format_time(seg.start)}]"

        if result.has_speakers and seg.speaker != current_speaker:
            current_speaker = seg.speaker
            speaker_label = seg.speaker or "Unknown"
            lines.append(f"\n**{speaker_label}** {timestamp}")
        else:
            lines.append(f"{timestamp} {seg.text.strip()}")
            continue

        lines.append(f"{seg.text.strip()}")

    return "\n".join(lines) + "\n"


def format_summary(summary_text: str) -> str:
    """Format a summary as markdown."""
    lines = ["# Meeting Summary\n", summary_text.strip(), ""]
    return "\n".join(lines) + "\n"
