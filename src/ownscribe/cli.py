"""CLI entry point for ownscribe."""

from __future__ import annotations

import os
import subprocess

import click

from ownscribe.config import Config, ensure_config_file


@click.group(invoke_without_command=True)
@click.option("--device", default=None, help="Audio input device name or index.")
@click.option("--no-summarize", is_flag=True, help="Skip LLM summarization.")
@click.option("--diarize", is_flag=True, help="Enable speaker diarization (needs HF token).")
@click.option("--format", "output_format", type=click.Choice(["markdown", "json"]), default=None, help="Output format.")
@click.option("--model", default=None, help="Whisper model size (tiny, base, small, medium, large-v3).")
@click.option("--mic", is_flag=True, help="Also capture microphone input (mixed with system audio).")
@click.option("--mic-device", default=None, help="Specific mic device name (implies --mic).")
@click.pass_context
def cli(
    ctx: click.Context,
    device: str | None,
    no_summarize: bool,
    diarize: bool,
    output_format: str | None,
    model: str | None,
    mic: bool,
    mic_device: str | None,
) -> None:
    """Fully local meeting transcription and summarization.

    Run without a subcommand to record, transcribe, and summarize a meeting.
    """
    ctx.ensure_object(dict)
    config = Config.load()

    # Apply CLI overrides
    if device is not None:
        config.audio.device = device
        if config.audio.backend == "coreaudio" and device:
            config.audio.backend = "sounddevice"
    if no_summarize:
        config.summarization.enabled = False
    if diarize:
        config.diarization.enabled = True
    if output_format:
        config.output.format = output_format
    if model:
        config.transcription.model = model
    if mic or mic_device:
        config.audio.mic = True
    if mic_device:
        config.audio.mic_device = mic_device

    ctx.obj["config"] = config

    if ctx.invoked_subcommand is None:
        from ownscribe.pipeline import run_pipeline
        run_pipeline(config)


@cli.command()
def devices() -> None:
    """List available audio input devices."""
    from ownscribe.audio.coreaudio import CoreAudioRecorder
    recorder = CoreAudioRecorder()
    if recorder.is_available():
        click.echo(recorder.list_devices())
    else:
        import sounddevice as sd
        click.echo("Available audio devices:\n")
        click.echo(sd.query_devices())


@cli.command()
@click.argument("file", type=click.Path(exists=True))
@click.option("--diarize", is_flag=True, help="Enable speaker diarization.")
@click.option("--model", default=None, help="Whisper model size.")
@click.option("--format", "output_format", type=click.Choice(["markdown", "json"]), default=None)
@click.pass_context
def transcribe(ctx: click.Context, file: str, diarize: bool, model: str | None, output_format: str | None) -> None:
    """Transcribe an audio file."""
    config = ctx.obj["config"]
    if diarize:
        config.diarization.enabled = True
    if model:
        config.transcription.model = model
    if output_format:
        config.output.format = output_format

    from ownscribe.pipeline import run_transcribe
    run_transcribe(config, file)


@cli.command()
@click.argument("file", type=click.Path(exists=True))
@click.pass_context
def summarize(ctx: click.Context, file: str) -> None:
    """Summarize a transcript file."""
    config = ctx.obj["config"]

    from ownscribe.pipeline import run_summarize
    run_summarize(config, file)


@cli.command()
def apps() -> None:
    """List running apps with PIDs for use with --pid."""
    from ownscribe.audio.coreaudio import CoreAudioRecorder
    recorder = CoreAudioRecorder()
    click.echo(recorder.list_apps())


@cli.command("config")
def config_cmd() -> None:
    """Open the configuration file in your editor."""
    path = ensure_config_file()
    editor = os.environ.get("EDITOR", "nano")
    click.echo(f"Opening {path} with {editor}...")
    subprocess.run([editor, str(path)])
