"""Tests for transcription helpers."""

from __future__ import annotations

import types
from unittest import mock

import pytest


class TestFfmpegCheck:
    def test_missing_ffmpeg_exits(self):
        from ownscribe.config import TranscriptionConfig
        from ownscribe.transcription.whisperx_transcriber import WhisperXTranscriber

        transcriber = WhisperXTranscriber(TranscriptionConfig(), None)

        with mock.patch("shutil.which", return_value=None), pytest.raises(SystemExit):
            transcriber.transcribe(mock.MagicMock())


class _FakeProgress:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []
        self.details: dict[str, str] = {}
        self.updates: list[tuple[str, float]] = []

    def begin(self, key: str) -> None:
        self.calls.append(("begin", key))

    def complete(self, key: str) -> None:
        self.calls.append(("complete", key))

    def fail(self, key: str) -> None:
        self.calls.append(("fail", key))

    def update(self, key: str, fraction: float) -> None:
        self.updates.append((key, fraction))

    def set_detail(self, key: str, text: str | None) -> None:
        if text is None:
            self.details.pop(key, None)
        else:
            self.details[key] = text

    def diarization_hook(self, step_name: str, _artifact, **kwargs) -> None:
        _ = (step_name, _artifact, kwargs)


class TestPrepareModels:
    def test_prepare_models_emits_preparing_models_lifecycle(self):
        from ownscribe.config import TranscriptionConfig
        from ownscribe.transcription.whisperx_transcriber import WhisperXTranscriber

        progress = _FakeProgress()
        transcriber = WhisperXTranscriber(TranscriptionConfig(language="en"), None, progress=progress)

        def passthrough(stage_label, fn, *args, **kwargs):
            _ = stage_label
            return fn(*args, **kwargs)

        with (
            mock.patch.object(transcriber, "_capture_prep_output", side_effect=passthrough),
            mock.patch.object(transcriber, "_load_model", side_effect=lambda: setattr(transcriber, "_model", object())),
            mock.patch.object(transcriber, "_load_align_model", return_value=(object(), object())),
        ):
            transcriber.prepare_models(language="en")

        assert ("begin", "preparing_models") in progress.calls
        assert ("complete", "preparing_models") in progress.calls
        assert ("fail", "preparing_models") not in progress.calls

    def test_prepare_models_skips_diarization_without_token(self):
        from ownscribe.config import DiarizationConfig, TranscriptionConfig
        from ownscribe.transcription.whisperx_transcriber import WhisperXTranscriber

        progress = _FakeProgress()
        diar = DiarizationConfig(enabled=True, hf_token="")
        transcriber = WhisperXTranscriber(TranscriptionConfig(language="en"), diar, progress=progress)

        def passthrough(stage_label, fn, *args, **kwargs):
            _ = stage_label
            return fn(*args, **kwargs)

        with (
            mock.patch.object(transcriber, "_capture_prep_output", side_effect=passthrough),
            mock.patch.object(transcriber, "_load_model", side_effect=lambda: setattr(transcriber, "_model", object())),
            mock.patch.object(transcriber, "_load_align_model", return_value=(object(), object())),
            mock.patch.object(transcriber, "_load_diarization_pipeline") as mock_diar_load,
        ):
            transcriber.prepare_models(language="en")

        mock_diar_load.assert_not_called()

    def test_prepare_models_reuses_loaded_whisper_model(self):
        from ownscribe.config import TranscriptionConfig
        from ownscribe.transcription.whisperx_transcriber import WhisperXTranscriber

        progress = _FakeProgress()
        transcriber = WhisperXTranscriber(TranscriptionConfig(language="en"), None, progress=progress)

        def passthrough(stage_label, fn, *args, **kwargs):
            _ = stage_label
            return fn(*args, **kwargs)

        with (
            mock.patch.object(transcriber, "_capture_prep_output", side_effect=passthrough),
            mock.patch.object(
                transcriber,
                "_load_model",
                side_effect=lambda: setattr(transcriber, "_model", object()),
            ) as mock_load_model,
            mock.patch.object(transcriber, "_load_align_model", return_value=(object(), object())),
        ):
            transcriber.prepare_models(language="en")
            transcriber.prepare_models(language="en")

        assert mock_load_model.call_count == 1


class TestDownloadProgressHooks:
    def test_on_download_progress_updates_detail_and_bar(self):
        from ownscribe.config import TranscriptionConfig
        from ownscribe.progress import DownloadProgressEvent
        from ownscribe.transcription.whisperx_transcriber import WhisperXTranscriber

        progress = _FakeProgress()
        transcriber = WhisperXTranscriber(TranscriptionConfig(), None, progress=progress)

        transcriber._on_download_progress(
            "preparing_models",
            "Loading Whisper model (base)",
            DownloadProgressEvent(filename="model.bin", percent=25.0),
        )

        assert ("preparing_models", 0.25) in progress.updates
        assert "Loading Whisper model (base)" in progress.details["preparing_models"]
        assert "model.bin" in progress.details["preparing_models"]
        assert "25%" not in progress.details["preparing_models"]

    def test_capture_download_output_resets_bar_to_zero(self):
        from ownscribe.config import TranscriptionConfig
        from ownscribe.transcription.whisperx_transcriber import WhisperXTranscriber

        progress = _FakeProgress()
        transcriber = WhisperXTranscriber(TranscriptionConfig(), None, progress=progress)

        def fake_loader():
            print("model.bin: 12%|##| 12MB/100MB [00:01<00:08]")

        transcriber._capture_download_output("preparing_models", "Loading Whisper model (base)", fake_loader)

        assert progress.updates
        assert progress.updates[0] == ("preparing_models", 0.0)
        assert any(key == "preparing_models" and frac > 0 for key, frac in progress.updates[1:])

    def test_transcribe_inner_does_not_use_preparing_models_step(self):
        from ownscribe.config import TranscriptionConfig
        from ownscribe.transcription.whisperx_transcriber import WhisperXTranscriber

        class _Audio:
            shape = (16000,)

        fake_whisperx = types.SimpleNamespace(
            load_audio=lambda _path: _Audio(),
            align=lambda *args, **kwargs: {"segments": []},
        )

        progress = _FakeProgress()
        transcriber = WhisperXTranscriber(TranscriptionConfig(language="en"), None, progress=progress)
        transcriber._model = mock.MagicMock()
        transcriber._model.transcribe.return_value = {"segments": [], "language": "en"}

        with (
            mock.patch.dict("sys.modules", {"whisperx": fake_whisperx}),
            mock.patch.object(transcriber, "_prepare_transcription_models") as mock_prepare_runtime,
            mock.patch.object(transcriber, "_load_align_model", return_value=(object(), object())),
            mock.patch.object(transcriber, "prepare_models") as mock_prepare_models,
        ):
            result = transcriber._transcribe_inner(mock.MagicMock())

        mock_prepare_models.assert_not_called()
        mock_prepare_runtime.assert_called_once_with(
            language="en",
            step_key="transcribing",
            show_deferred_align_note=False,
        )
        assert ("begin", "transcribing") in progress.calls
        assert ("begin", "preparing_models") not in progress.calls
        assert result.language == "en"
