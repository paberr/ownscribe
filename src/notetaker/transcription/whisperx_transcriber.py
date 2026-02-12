"""WhisperX-based transcription with optional diarization."""

from __future__ import annotations

from pathlib import Path

import click

from notetaker.config import DiarizationConfig, TranscriptionConfig
from notetaker.transcription.base import Transcriber
from notetaker.transcription.models import Segment, TranscriptResult, Word


class WhisperXTranscriber(Transcriber):
    """Transcribes audio using WhisperX (faster-whisper + optional pyannote diarization)."""

    def __init__(
        self,
        transcription_config: TranscriptionConfig,
        diarization_config: DiarizationConfig | None = None,
    ) -> None:
        self._tx_config = transcription_config
        self._diar_config = diarization_config
        self._model = None

    def _load_model(self):
        import os

        os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

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
        import whisperx

        if self._model is None:
            click.echo("Loading WhisperX model...")
            self._load_model()

        click.echo(f"Transcribing {audio_path.name}...")
        audio = whisperx.load_audio(str(audio_path))
        result = self._model.transcribe(audio, batch_size=16)

        language = result.get("language", "")

        # Alignment for word-level timestamps
        click.echo("Aligning transcript...")
        align_model, align_metadata = whisperx.load_align_model(
            language_code=language, device="cpu"
        )
        result = whisperx.align(
            result["segments"],
            align_model,
            align_metadata,
            audio,
            device="cpu",
            return_char_alignments=False,
        )

        # Optional diarization
        if (
            self._diar_config
            and self._diar_config.enabled
            and self._diar_config.hf_token
        ):
            click.echo("Running speaker diarization...")
            diarize_model = whisperx.DiarizationPipeline(
                use_auth_token=self._diar_config.hf_token, device="cpu"
            )
            diarize_kwargs = {}
            if self._diar_config.min_speakers > 0:
                diarize_kwargs["min_speakers"] = self._diar_config.min_speakers
            if self._diar_config.max_speakers > 0:
                diarize_kwargs["max_speakers"] = self._diar_config.max_speakers

            diarize_segments = diarize_model(audio, **diarize_kwargs)
            result = whisperx.assign_word_speakers(diarize_segments, result)
        elif self._diar_config and self._diar_config.enabled:
            click.echo(
                "Diarization requested but no HF token configured. "
                "Set HF_TOKEN env var or hf_token in config. Skipping."
            )

        # Convert to our data models
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

        duration = audio.shape[0] / 16000.0  # whisperx loads at 16kHz
        return TranscriptResult(segments=segments, language=language, duration=duration)
