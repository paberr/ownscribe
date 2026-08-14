"""Tests for JSON output formatter."""

import json

from ownscribe.output.json_output import format_transcript_json, parse_transcript_json
from ownscribe.transcription.models import Segment, TranscriptResult, Word


class TestFormatTranscriptJson:
    def test_valid_json(self, sample_transcript):
        output = format_transcript_json(sample_transcript)
        parsed = json.loads(output)
        assert isinstance(parsed, dict)

    def test_all_fields_present(self, sample_transcript):
        output = format_transcript_json(sample_transcript)
        parsed = json.loads(output)
        assert "segments" in parsed
        assert "language" in parsed
        assert "duration" in parsed
        assert len(parsed["segments"]) == 2
        seg = parsed["segments"][0]
        assert "text" in seg
        assert "start" in seg
        assert "end" in seg

    def test_round_trip_with_speakers(self, diarized_transcript):
        parsed = parse_transcript_json(format_transcript_json(diarized_transcript))

        assert parsed is not None
        assert parsed.speaker_text == diarized_transcript.speaker_text
        assert parsed.language == "en"
        assert parsed.duration == 6.0

    def test_round_trip_keeps_words(self):
        result = TranscriptResult(
            segments=[
                Segment(
                    text="Hello there.",
                    start=0.0,
                    end=1.0,
                    speaker="SPEAKER_00",
                    words=[Word(text="Hello", start=0.0, end=0.5, speaker="SPEAKER_00", score=0.9)],
                )
            ],
            language="en",
        )

        parsed = parse_transcript_json(format_transcript_json(result))

        assert parsed is not None
        assert parsed.segments[0].words[0].text == "Hello"
        assert parsed.segments[0].words[0].score == 0.9

    def test_returns_none_for_non_transcript_json(self):
        assert parse_transcript_json("not json at all") is None
        assert parse_transcript_json('{"unrelated": 1}') is None
        assert parse_transcript_json("[1, 2, 3]") is None

    def test_non_ascii_preserved(self):
        result = TranscriptResult(
            segments=[Segment(text="Tsch\u00fcss und auf Wiedersehen!", start=0.0, end=2.0)],
            language="de",
        )
        output = format_transcript_json(result)
        assert "Tsch\u00fcss" in output
        # Ensure it's not escaped to \\u
        parsed = json.loads(output)
        assert parsed["segments"][0]["text"] == "Tsch\u00fcss und auf Wiedersehen!"
