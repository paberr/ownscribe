"""WhisperX-based transcription with optional diarization."""

from __future__ import annotations

import contextlib
import logging
import os
import warnings
from pathlib import Path

import click

from ownscribe.config import DiarizationConfig, TranscriptionConfig
from ownscribe.progress import NullProgress, ProgressWriter
from ownscribe.transcription.base import Transcriber
from ownscribe.transcription.models import Segment, TranscriptResult, Word

_SAMPLE_RATE = 16000


class WhisperXTranscriber(Transcriber):
    """Transcribes audio using WhisperX (faster-whisper + optional pyannote diarization)."""

    def __init__(
        self,
        transcription_config: TranscriptionConfig,
        diarization_config: DiarizationConfig | None = None,
        progress: NullProgress | None = None,
    ) -> None:
        self._tx_config = transcription_config
        self._diar_config = diarization_config
        self._progress = progress or NullProgress()
        self._model = None

    def _load_model(self):
        import whisperx

        device = "cpu"
        compute_type = "int8"
        self._model = whisperx.load_model(
            self._tx_config.model,
            device,
            compute_type=compute_type,
            language=self._tx_config.language or None,
        )

    def transcribe(self, audio_path: Path) -> TranscriptResult:
        # --- Telemetry toggle (must happen before importing whisperx) ---
        os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
        if self._diar_config is None or not self._diar_config.telemetry:
            os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

        hf_token_warning: str | None = None
        if (
            self._diar_config
            and self._diar_config.enabled
            and not self._diar_config.hf_token
        ):
            hf_token_warning = (
                "Diarization requested but no HF token configured. "
                "Set HF_TOKEN env var or hf_token in config. Skipping."
            )

        # Suppress all noise from whisperx / pyannote / torch / lightning
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Suppress named loggers that bypass root (whisperx has propagate=False)
            for name in ("whisperx", "lightning", "pytorch_lightning"):
                logging.getLogger(name).setLevel(logging.WARNING)
            result = self._transcribe_inner(audio_path)

        if hf_token_warning:
            click.echo(hf_token_warning, err=True)

        return result

    def _transcribe_inner(self, audio_path: Path) -> TranscriptResult:
        import whisperx

        progress = self._progress

        devnull = open(os.devnull, "w")  # noqa: SIM115
        try:
            # Outer redirect: catch all stray print() from pyannote/lightning
            with contextlib.redirect_stdout(devnull):
                progress.begin("transcribing")

                if self._model is None:
                    with contextlib.redirect_stderr(devnull):
                        self._load_model()

                audio = whisperx.load_audio(str(audio_path))

                tx_writer = ProgressWriter(
                    lambda frac: progress.update("transcribing", frac),
                    offset=0.0, scale=0.5,
                )
                align_writer = ProgressWriter(
                    lambda frac: progress.update("transcribing", frac),
                    offset=0.5, scale=0.5,
                )

                # Nested redirect overrides devnull → captures progress
                with contextlib.redirect_stdout(tx_writer):
                    result = self._model.transcribe(
                        audio, batch_size=16, print_progress=True, combined_progress=True
                    )

                language = result.get("language", "")

                align_model, align_metadata = whisperx.load_align_model(
                    language_code=language, device="cpu"
                )
                with contextlib.redirect_stdout(align_writer):
                    result = whisperx.align(
                        result["segments"],
                        align_model,
                        align_metadata,
                        audio,
                        device="cpu",
                        return_char_alignments=False,
                        print_progress=True,
                        combined_progress=True,
                    )

                progress.complete("transcribing")

                # --- Optional diarization ---
                if (
                    self._diar_config
                    and self._diar_config.enabled
                    and self._diar_config.hf_token
                ):
                    result = self._diarize(audio, result, devnull)
        finally:
            devnull.close()

        # --- Convert to our data models ---
        segments = []
        for seg in result.get("segments", []):
            words = []
            for w in seg.get("words", []):
                words.append(
                    Word(
                        text=w.get("word", ""),
                        start=w.get("start", 0.0),
                        end=w.get("end", 0.0),
                        speaker=w.get("speaker"),
                        score=w.get("score", 0.0),
                    )
                )
            segments.append(
                Segment(
                    text=seg.get("text", ""),
                    start=seg.get("start", 0.0),
                    end=seg.get("end", 0.0),
                    speaker=seg.get("speaker"),
                    words=words,
                )
            )

        duration = audio.shape[0] / float(_SAMPLE_RATE)
        return TranscriptResult(segments=segments, language=language, duration=duration)

    def _diarize(self, audio, result, devnull):
        import pandas as pd
        import torch
        import whisperx
        from whisperx.diarize import DiarizationPipeline

        progress = self._progress
        progress.begin("diarizing")

        # Load the diarization pipeline (model loading happens inside)
        with contextlib.redirect_stderr(devnull):
            diarize_model = DiarizationPipeline(
                use_auth_token=self._diar_config.hf_token, device="cpu"
            )

        # Build audio_data dict the same way whisperx does internally
        audio_data = {
            "waveform": torch.from_numpy(audio[None, :]),
            "sample_rate": _SAMPLE_RATE,
        }

        diarize_kwargs = {}
        if self._diar_config.min_speakers > 0:
            diarize_kwargs["min_speakers"] = self._diar_config.min_speakers
        if self._diar_config.max_speakers > 0:
            diarize_kwargs["max_speakers"] = self._diar_config.max_speakers

        # Call pyannote pipeline directly with progress hook
        diarization = diarize_model.model(
            audio_data, hook=progress.diarization_hook, **diarize_kwargs
        )

        progress.complete("diarizing")

        # Convert to DataFrame (replicating whisperx/diarize.py logic)
        diarize_df = pd.DataFrame(
            diarization.itertracks(yield_label=True),
            columns=["segment", "label", "speaker"],
        )
        diarize_df["start"] = diarize_df["segment"].apply(lambda x: x.start)
        diarize_df["end"] = diarize_df["segment"].apply(lambda x: x.end)

        return whisperx.assign_word_speakers(diarize_df, result)
