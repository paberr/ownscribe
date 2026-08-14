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


class TestSpeakerText:
    def test_prefixes_on_speaker_change(self, diarized_transcript):
        assert diarized_transcript.speaker_text == (
            "SPEAKER_00: Hi, let's get started.\n"
            "SPEAKER_01: Sounds good.\n"
            "SPEAKER_00: First topic is the budget."
        )

    def test_joins_consecutive_segments_of_same_speaker(self):
        result = TranscriptResult(
            segments=[
                Segment(text="Hello.", start=0.0, end=1.0, speaker="SPEAKER_00"),
                Segment(text="Let's begin.", start=1.0, end=2.0, speaker="SPEAKER_00"),
                Segment(text="Sure.", start=2.0, end=3.0, speaker="SPEAKER_01"),
            ]
        )
        assert result.speaker_text == "SPEAKER_00: Hello. Let's begin.\nSPEAKER_01: Sure."

    def test_falls_back_to_full_text_without_speakers(self, sample_transcript):
        assert sample_transcript.speaker_text == sample_transcript.full_text

    def test_labels_missing_speaker_as_unknown(self):
        result = TranscriptResult(
            segments=[
                Segment(text="Anyone there?", start=0.0, end=1.0),
                Segment(text="Yes.", start=1.0, end=2.0, speaker="SPEAKER_01"),
            ]
        )
        assert result.speaker_text == "Unknown: Anyone there?\nSPEAKER_01: Yes."

    def test_skips_empty_segments(self):
        result = TranscriptResult(
            segments=[
                Segment(text="  ", start=0.0, end=1.0, speaker="SPEAKER_00"),
                Segment(text="Real text.", start=1.0, end=2.0, speaker="SPEAKER_00"),
            ]
        )
        assert result.speaker_text == "SPEAKER_00: Real text."

    def test_empty_segments_list(self):
        assert TranscriptResult(segments=[]).speaker_text == ""


class TestHasSpeakers:
    def test_true_when_speakers_present(self, diarized_transcript):
        assert diarized_transcript.has_speakers is True

    def test_false_when_no_speakers(self, sample_transcript):
        assert sample_transcript.has_speakers is False

    def test_false_for_empty(self):
        result = TranscriptResult(segments=[])
        assert result.has_speakers is False
