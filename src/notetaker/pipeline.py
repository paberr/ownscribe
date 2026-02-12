"""Pipeline orchestration: record -> transcribe -> summarize -> output."""

from __future__ import annotations

import signal
import sys
import time
from datetime import datetime
from pathlib import Path

import click

from notetaker.config import Config


def _check_audio_silence(audio_path: Path) -> None:
    """Check if the recorded audio is silent and warn the user."""
    try:
        import numpy as np
        import soundfile as sf
    except ImportError:
        return  # Skip check if deps not available

    try:
        # Read up to 5 seconds
        info = sf.info(audio_path)
        frames_to_read = min(int(info.samplerate * 5), info.frames)
        data, _ = sf.read(audio_path, frames=frames_to_read, dtype="float32")
        peak = float(np.max(np.abs(data)))
    except Exception:
        return  # Don't block pipeline on check failure

    if peak < 1e-6:
        click.echo(
            "\nError: Recorded audio is completely silent (peak amplitude ~0).\n"
            "This usually means Screen Recording permission is missing.\n"
            "Fix: System Settings > Privacy & Security > Screen Recording "
            "— enable your terminal app, then restart it.\n",
            err=True,
        )
        raise SystemExit(1)


def _get_output_dir(config: Config) -> Path:
    """Create and return a timestamped output directory."""
    base = config.output.resolved_dir
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out_dir = base / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _create_recorder(config: Config):
    """Create the appropriate audio recorder based on config."""
    if config.audio.backend == "coreaudio" and not config.audio.device:
        from notetaker.audio.coreaudio import CoreAudioRecorder

        recorder = CoreAudioRecorder(mic=config.audio.mic, mic_device=config.audio.mic_device)
        if recorder.is_available():
            return recorder
        click.echo("Core Audio helper not found, falling back to sounddevice.")

    from notetaker.audio.sounddevice_recorder import SoundDeviceRecorder

    device = config.audio.device or None
    # Try to parse as int (device index)
    if isinstance(device, str) and device.isdigit():
        device = int(device)
    return SoundDeviceRecorder(device=device)


def _create_transcriber(config: Config):
    """Create the WhisperX transcriber."""
    from notetaker.transcription.whisperx_transcriber import WhisperXTranscriber

    diar_config = config.diarization if config.diarization.enabled else None
    return WhisperXTranscriber(config.transcription, diar_config)


def _create_summarizer(config: Config):
    """Create the appropriate summarizer based on config."""
    if config.summarization.backend == "openai":
        from notetaker.summarization.openai_summarizer import OpenAISummarizer
        return OpenAISummarizer(config.summarization)
    else:
        from notetaker.summarization.ollama_summarizer import OllamaSummarizer
        return OllamaSummarizer(config.summarization)


def _format_output(config: Config, transcript_result, summary_text: str | None = None) -> tuple[str, str | None]:
    """Format transcript and optional summary. Returns (transcript_str, summary_str)."""
    if config.output.format == "json":
        from notetaker.output.json_output import format_transcript_json
        return format_transcript_json(transcript_result), summary_text
    else:
        from notetaker.output.markdown import format_transcript, format_summary
        tx = format_transcript(transcript_result)
        sm = format_summary(summary_text) if summary_text else None
        return tx, sm


def run_pipeline(config: Config) -> None:
    """Run the full pipeline: record, transcribe, summarize, output."""
    out_dir = _get_output_dir(config)
    audio_path = out_dir / "recording.wav"

    # 1. Record
    recorder = _create_recorder(config)
    click.echo("Starting recording... Press Ctrl+C to stop.\n")
    recorder.start(audio_path)

    start_time = time.time()
    stop_event = False

    def on_interrupt(sig, frame):
        nonlocal stop_event
        stop_event = True

    original_handler = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, on_interrupt)

    warned_no_data = False
    try:
        while not stop_event:
            elapsed = time.time() - start_time
            mins, secs = divmod(int(elapsed), 60)
            click.echo(f"\r  Recording: {mins:02d}:{secs:02d}", nl=False)
            if (
                not warned_no_data
                and elapsed >= 3
                and audio_path.exists()
                and audio_path.stat().st_size <= 44
            ):
                click.echo(
                    "\n\n  Warning: No audio data received yet.\n",
                    err=True,
                )
                warned_no_data = True
            time.sleep(0.5)
    finally:
        signal.signal(signal.SIGINT, original_handler)

    click.echo("\n\nStopping recording...")
    recorder.stop()

    if not audio_path.exists() or audio_path.stat().st_size <= 44:
        click.echo(
            "Error: No audio was captured. Make sure audio is playing on your system.",
            err=True,
        )
        raise SystemExit(1)

    click.echo(f"Audio saved to {audio_path}\n")

    # Check for silent audio before spending time on transcription
    _check_audio_silence(audio_path)

    # 2. Transcribe
    _do_transcribe_and_summarize(config, audio_path, out_dir)


def run_transcribe(config: Config, audio_file: str) -> None:
    """Transcribe an audio file and output the result."""
    audio_path = Path(audio_file)
    _check_audio_silence(audio_path)
    out_dir = _get_output_dir(config)
    _do_transcribe_and_summarize(config, audio_path, out_dir, summarize=False)


def run_summarize(config: Config, transcript_file: str) -> None:
    """Summarize a transcript file."""
    transcript_text = Path(transcript_file).read_text()

    summarizer = _create_summarizer(config)
    if not summarizer.is_available():
        click.echo(
            f"Error: {config.summarization.backend} is not reachable at {config.summarization.host}. "
            "Is the server running?",
            err=True,
        )
        raise SystemExit(1)

    summary = summarizer.summarize(transcript_text)

    from notetaker.output.markdown import format_summary

    out_dir = _get_output_dir(config)
    summary_md = format_summary(summary)
    summary_path = out_dir / "summary.md"
    summary_path.write_text(summary_md)

    click.echo(f"\n{summary_md}")
    click.echo(f"Summary saved to {summary_path}")


def _do_transcribe_and_summarize(
    config: Config,
    audio_path: Path,
    out_dir: Path,
    summarize: bool = True,
) -> None:
    """Shared logic for transcribe + optional summarize."""
    try:
        transcriber = _create_transcriber(config)
    except ImportError:
        click.echo(
            "Error: WhisperX is not installed. Install with:\n"
            "  uv pip install 'notetaker[transcription]'",
            err=True,
        )
        raise SystemExit(1)

    result = transcriber.transcribe(audio_path)

    transcript_str, _ = _format_output(config, result)

    ext = "json" if config.output.format == "json" else "md"
    transcript_path = out_dir / f"transcript.{ext}"
    transcript_path.write_text(transcript_str)
    click.echo(f"Transcript saved to {transcript_path}")

    # 3. Summarize
    if summarize and config.summarization.enabled:
        summarizer = _create_summarizer(config)
        if not summarizer.is_available():
            click.echo(
                f"\nWarning: {config.summarization.backend} is not reachable at {config.summarization.host}. "
                "Skipping summarization. Is the server running?",
                err=True,
            )
            return

        summary = summarizer.summarize(result.full_text)
        _, summary_str = _format_output(config, result, summary)

        summary_path = out_dir / f"summary.{ext}"
        summary_path.write_text(summary_str or summary)
        click.echo(f"Summary saved to {summary_path}")
        click.echo(f"\n{summary_str or summary}")
    elif not summarize:
        click.echo(f"\n{transcript_str}")
