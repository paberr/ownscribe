"""Tests for JSON output formatter."""

import json

from ownscribe.output.json_output import format_transcript_json
from ownscribe.transcription.models import Segment, TranscriptResult


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
