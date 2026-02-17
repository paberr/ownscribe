"""Tests for pipeline orchestration."""

from __future__ import annotations

import json
from unittest import mock

from ownscribe.config import Config
from ownscribe.transcription.models import Segment, TranscriptResult


class TestCreateRecorder:
    def test_coreaudio_when_available(self):
        from ownscribe.pipeline import _create_recorder

        config = Config()
        config.audio.backend = "coreaudio"
        config.audio.device = ""

        with mock.patch("ownscribe.audio.coreaudio.CoreAudioRecorder") as mock_cls:
            mock_cls.return_value.is_available.return_value = True
            recorder = _create_recorder(config)
            assert recorder == mock_cls.return_value

    def test_fallback_to_sounddevice(self):
        from ownscribe.pipeline import _create_recorder

        config = Config()
        config.audio.backend = "coreaudio"
        config.audio.device = ""

        with (
            mock.patch("ownscribe.audio.coreaudio.CoreAudioRecorder") as mock_ca,
            mock.patch("ownscribe.audio.sounddevice_recorder.SoundDeviceRecorder") as mock_sd,
        ):
            mock_ca.return_value.is_available.return_value = False
            recorder = _create_recorder(config)
            assert recorder == mock_sd.return_value

    def test_sounddevice_when_device_set(self):
        from ownscribe.pipeline import _create_recorder

        config = Config()
        config.audio.backend = "coreaudio"
        config.audio.device = "USB Mic"

        with mock.patch("ownscribe.audio.sounddevice_recorder.SoundDeviceRecorder") as mock_sd:
            recorder = _create_recorder(config)
            assert recorder == mock_sd.return_value


class TestFormatOutput:
    def test_markdown_format(self, sample_transcript):
        from ownscribe.pipeline import _format_output

        config = Config()
        config.output.format = "markdown"

        transcript_str, summary_str = _format_output(config, sample_transcript)
        assert "# Transcript" in transcript_str
        assert summary_str is None

    def test_markdown_with_summary(self, sample_transcript):
        from ownscribe.pipeline import _format_output

        config = Config()
        config.output.format = "markdown"

        transcript_str, summary_str = _format_output(config, sample_transcript, "A great meeting.")
        assert "# Transcript" in transcript_str
        assert "# Meeting Summary" in summary_str
        assert "A great meeting." in summary_str

    def test_json_format(self, sample_transcript):
        from ownscribe.pipeline import _format_output

        config = Config()
        config.output.format = "json"

        transcript_str, _summary_str = _format_output(config, sample_transcript)
        parsed = json.loads(transcript_str)
        assert "segments" in parsed


class TestSlugify:
    def test_basic(self):
        from ownscribe.pipeline import _slugify

        assert _slugify("Q3 Budget Planning Review") == "q3-budget-planning-review"

    def test_strips_special_chars(self):
        from ownscribe.pipeline import _slugify

        assert _slugify("Hello, World! @#$") == "hello-world"

    def test_truncates_to_max_length(self):
        from ownscribe.pipeline import _slugify

        result = _slugify("a " * 100, max_length=10)
        assert len(result) <= 10

    def test_empty_input(self):
        from ownscribe.pipeline import _slugify

        assert _slugify("") == ""

    def test_colons_removed(self):
        from ownscribe.pipeline import _slugify

        assert _slugify("Meeting: Budget Review") == "meeting-budget-review"


class TestGenerateTitleSlug:
    def test_returns_slug(self):
        from ownscribe.pipeline import _generate_title_slug

        mock_summarizer = mock.MagicMock()
        mock_summarizer.generate_title.return_value = "Budget Review"

        assert _generate_title_slug("summary text", mock_summarizer) == "budget-review"

    def test_returns_empty_on_empty_slug(self):
        from ownscribe.pipeline import _generate_title_slug

        mock_summarizer = mock.MagicMock()
        mock_summarizer.generate_title.return_value = "!!!"  # slugifies to empty

        assert _generate_title_slug("summary", mock_summarizer) == ""

    def test_returns_empty_on_llm_failure(self):
        from ownscribe.pipeline import _generate_title_slug

        mock_summarizer = mock.MagicMock()
        mock_summarizer.generate_title.side_effect = Exception("LLM down")

        assert _generate_title_slug("summary", mock_summarizer) == ""


class TestDoTranscribeAndSummarize:
    """Test _do_transcribe_and_summarize with mocked transcriber/summarizer."""

    def _make_transcript(self) -> TranscriptResult:
        return TranscriptResult(
            segments=[Segment(text="Hello world.", start=0.0, end=1.5)],
            language="en",
            duration=1.5,
        )

    def test_transcribe_only(self, tmp_path):
        from ownscribe.pipeline import _do_transcribe_and_summarize

        config = Config()
        config.output.format = "markdown"
        audio_path = tmp_path / "recording.wav"
        audio_path.touch()

        mock_transcriber = mock.MagicMock()
        mock_transcriber.transcribe.return_value = self._make_transcript()

        with mock.patch("ownscribe.pipeline._create_transcriber", return_value=mock_transcriber):
            _do_transcribe_and_summarize(config, audio_path, tmp_path, summarize=False)

        assert (tmp_path / "transcript.md").exists()
        assert not (tmp_path / "summary.md").exists()

    def test_transcribe_and_summarize(self, tmp_path):
        from ownscribe.pipeline import _do_transcribe_and_summarize

        config = Config()
        config.output.format = "markdown"
        config.summarization.enabled = True
        audio_path = tmp_path / "recording.wav"
        audio_path.touch()

        mock_transcriber = mock.MagicMock()
        mock_transcriber.transcribe.return_value = self._make_transcript()

        mock_summarizer = mock.MagicMock()
        mock_summarizer.is_available.return_value = True
        mock_summarizer.summarize.return_value = "## Summary\nGood meeting."

        with (
            mock.patch("ownscribe.pipeline._create_transcriber", return_value=mock_transcriber),
            mock.patch("ownscribe.pipeline._create_summarizer", return_value=mock_summarizer),
        ):
            _do_transcribe_and_summarize(config, audio_path, tmp_path, summarize=True)

        assert (tmp_path / "transcript.md").exists()
        assert (tmp_path / "summary.md").exists()
        assert "Summary" in (tmp_path / "summary.md").read_text()

    def test_summarizer_unavailable_skips_gracefully(self, tmp_path):
        from ownscribe.pipeline import _do_transcribe_and_summarize

        config = Config()
        config.output.format = "markdown"
        config.summarization.enabled = True
        audio_path = tmp_path / "recording.wav"
        audio_path.touch()

        mock_transcriber = mock.MagicMock()
        mock_transcriber.transcribe.return_value = self._make_transcript()

        mock_summarizer = mock.MagicMock()
        mock_summarizer.is_available.return_value = False

        with (
            mock.patch("ownscribe.pipeline._create_transcriber", return_value=mock_transcriber),
            mock.patch("ownscribe.pipeline._create_summarizer", return_value=mock_summarizer),
        ):
            _do_transcribe_and_summarize(config, audio_path, tmp_path, summarize=True)

        assert (tmp_path / "transcript.md").exists()
        assert not (tmp_path / "summary.md").exists()

    def test_json_output_format(self, tmp_path):
        from ownscribe.pipeline import _do_transcribe_and_summarize

        config = Config()
        config.output.format = "json"
        audio_path = tmp_path / "recording.wav"
        audio_path.touch()

        mock_transcriber = mock.MagicMock()
        mock_transcriber.transcribe.return_value = self._make_transcript()

        with mock.patch("ownscribe.pipeline._create_transcriber", return_value=mock_transcriber):
            _do_transcribe_and_summarize(config, audio_path, tmp_path, summarize=False)

        assert (tmp_path / "transcript.json").exists()
        assert not (tmp_path / "transcript.md").exists()

    def test_keep_recording_false_deletes_wav(self, tmp_path):
        from ownscribe.pipeline import _do_transcribe_and_summarize

        config = Config()
        config.output.format = "markdown"
        config.output.keep_recording = False
        audio_path = tmp_path / "recording.wav"
        audio_path.write_bytes(b"fake audio data")

        mock_transcriber = mock.MagicMock()
        mock_transcriber.transcribe.return_value = self._make_transcript()

        with mock.patch("ownscribe.pipeline._create_transcriber", return_value=mock_transcriber):
            _do_transcribe_and_summarize(config, audio_path, tmp_path, summarize=False)

        assert (tmp_path / "transcript.md").exists()
        assert not audio_path.exists()

    def test_keep_recording_true_keeps_wav(self, tmp_path):
        from ownscribe.pipeline import _do_transcribe_and_summarize

        config = Config()
        config.output.format = "markdown"
        config.output.keep_recording = True
        audio_path = tmp_path / "recording.wav"
        audio_path.write_bytes(b"fake audio data")

        mock_transcriber = mock.MagicMock()
        mock_transcriber.transcribe.return_value = self._make_transcript()

        with mock.patch("ownscribe.pipeline._create_transcriber", return_value=mock_transcriber):
            _do_transcribe_and_summarize(config, audio_path, tmp_path, summarize=False)

        assert (tmp_path / "transcript.md").exists()
        assert audio_path.exists()
