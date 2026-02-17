"""Tests for transcription helpers."""

from __future__ import annotations

from unittest import mock

import pytest


class TestFfmpegCheck:
    def test_missing_ffmpeg_exits(self):
        from ownscribe.config import TranscriptionConfig
        from ownscribe.transcription.whisperx_transcriber import WhisperXTranscriber

        transcriber = WhisperXTranscriber(TranscriptionConfig(), None)

        with mock.patch("shutil.which", return_value=None), pytest.raises(SystemExit):
            transcriber.transcribe(mock.MagicMock())
