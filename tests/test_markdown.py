"""Tests for markdown output formatter."""

from ownscribe.output.markdown import _format_time, format_summary, format_transcript


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


class TestFormatSummary:
    def test_wraps_text(self):
        output = format_summary("This is the summary.")
        assert output.startswith("# Meeting Summary\n")
        assert "This is the summary." in output

    def test_strips_whitespace(self):
        output = format_summary("  extra spaces  ")
        assert "extra spaces" in output
        assert "  extra" not in output
