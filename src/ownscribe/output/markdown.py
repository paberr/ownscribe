"""Markdown output formatter for transcripts."""

from __future__ import annotations

import re

from ownscribe.transcription.models import Segment, TranscriptResult

# Inverses of format_transcript's line shapes: a speaker header introducing a
# turn, and a plain timestamped line when diarization did not run.
_TIME = r"(?:(?P<h>\d{1,2}):)?(?P<m>\d{1,2}):(?P<s>\d{2})"
_SPEAKER_LINE_RE = re.compile(rf"^\*\*(?P<speaker>[^*]+)\*\*\s+\[{_TIME}\]\s*$")
_STAMPED_LINE_RE = re.compile(rf"^\[{_TIME}\]\s*(?P<text>.*)$")
_METADATA_RE = re.compile(r"^\*\*(?:Language|Duration):\*\*")


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


def _parse_time(match: re.Match) -> float:
    """Seconds from an [HH:MM:SS] or [MM:SS] stamp matched by _TIME."""
    return int(match.group("h") or 0) * 3600 + int(match.group("m")) * 60 + int(match.group("s"))


def parse_transcript(text: str) -> TranscriptResult | None:
    """Read back a transcript written by format_transcript.

    Drops the heading and timestamps and restores the speaker on each segment.
    Returns None when *text* holds no recognisable transcript lines, so callers
    can fall back to treating the file as plain text.
    """
    segments: list[Segment] = []
    language = ""
    speaker: str | None = None

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("# "):
            continue

        if line.startswith("**Language:**"):
            language = line.removeprefix("**Language:**").strip()
            continue
        if _METADATA_RE.match(line):
            continue

        if m := _SPEAKER_LINE_RE.match(line):
            speaker = m.group("speaker").strip()
            segments.append(Segment(text="", start=_parse_time(m), end=0.0, speaker=speaker))
            continue

        if m := _STAMPED_LINE_RE.match(line):
            segments.append(Segment(text=m.group("text").strip(), start=_parse_time(m), end=0.0))
            continue

        # Body line following a speaker header.
        if segments and segments[-1].speaker is not None and not segments[-1].text:
            segments[-1].text = line
        else:
            segments.append(Segment(text=line, start=0.0, end=0.0, speaker=speaker))

    segments = [seg for seg in segments if seg.text]
    if not segments:
        return None
    return TranscriptResult(segments=segments, language=language)


def format_summary(summary_text: str) -> str:
    """Format a summary as markdown."""
    lines = ["# Meeting Summary\n", summary_text.strip(), ""]
    return "\n".join(lines) + "\n"
