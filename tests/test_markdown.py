"""Tests for markdown output formatter."""

from ownscribe.output.markdown import (
    _format_time,
    format_summary,
    format_transcript,
    parse_transcript,
)


class TestFormatTime:
    def test_zero_seconds(self):
        assert _format_time(0) == "00:00"

    def test_65_seconds(self):
        assert _format_time(65) == "01:05"

    def test_3661_seconds(self):
        assert _format_time(3661) == "01:01:01"

    def test_exact_hour(self):
        assert _format_time(3600) == "01:00:00"


class TestFormatTranscript:
    def test_without_speakers(self, sample_transcript):
        output = format_transcript(sample_transcript)
        assert output.startswith("# Transcript\n")
        assert "**Language:** en" in output
        assert "[00:00] Hello world." in output
        assert "[00:01] How are you?" in output

    def test_with_speakers(self, diarized_transcript):
        output = format_transcript(diarized_transcript)
        assert "**SPEAKER_00**" in output
        assert "**SPEAKER_01**" in output
        assert "Hi, let's get started." in output


class TestParseTranscript:
    """format_transcript's inverse, so resume/summarize never feed the LLM markdown."""

    def test_round_trip_without_speakers(self, sample_transcript):
        parsed = parse_transcript(format_transcript(sample_transcript))

        assert parsed is not None
        assert [seg.text for seg in parsed.segments] == ["Hello world.", "How are you?"]
        assert parsed.language == "en"

    def test_round_trip_with_speakers(self, diarized_transcript):
        parsed = parse_transcript(format_transcript(diarized_transcript))

        assert parsed is not None
        assert parsed.speaker_text == diarized_transcript.speaker_text

    def test_drops_timestamps_and_heading(self, sample_transcript):
        parsed = parse_transcript(format_transcript(sample_transcript))

        assert parsed is not None
        assert "[00:00]" not in parsed.speaker_text
        assert "# Transcript" not in parsed.speaker_text
        assert "**Duration:**" not in parsed.speaker_text

    def test_parses_hour_long_timestamps(self):
        parsed = parse_transcript("# Transcript\n\n[01:02:03] Still going.\n")

        assert parsed is not None
        assert parsed.segments[0].start == 3723
        assert parsed.segments[0].text == "Still going."

    def test_returns_none_for_plain_text(self):
        assert parse_transcript("") is None
        assert parse_transcript("# Transcript\n\n**Duration:** 00:10\n") is None


class TestFormatSummary:
    def test_wraps_text(self):
        output = format_summary("This is the summary.")
        assert output.startswith("# Meeting Summary\n")
        assert "This is the summary." in output

    def test_strips_whitespace(self):
        output = format_summary("  extra spaces  ")
        assert "extra spaces" in output
        assert "  extra" not in output
