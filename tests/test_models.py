"""Tests for transcription data models."""

from ownscribe.transcription.models import Segment, TranscriptResult


class TestFullText:
    def test_joins_segments(self, sample_transcript):
        assert sample_transcript.full_text == "Hello world. How are you?"

    def test_empty_segments(self):
        result = TranscriptResult(segments=[])
        assert result.full_text == ""

    def test_strips_whitespace(self):
        result = TranscriptResult(
            segments=[
                Segment(text="  padded  ", start=0.0, end=1.0),
                Segment(text=" text ", start=1.0, end=2.0),
            ]
        )
        assert result.full_text == "padded text"


class TestHasSpeakers:
    def test_true_when_speakers_present(self, diarized_transcript):
        assert diarized_transcript.has_speakers is True

    def test_false_when_no_speakers(self, sample_transcript):
        assert sample_transcript.has_speakers is False

    def test_false_for_empty(self):
        result = TranscriptResult(segments=[])
        assert result.has_speakers is False
