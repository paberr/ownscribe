"""CLI entry point for ownscribe."""

from __future__ import annotations

import os
import shutil
import subprocess

import click

from ownscribe.config import CONFIG_DIR, Config, ensure_config_file

# Canonical paths for cleanup
_CACHE_DIR = os.path.expanduser("~/.local/share/ownscribe")
_CONFIG_DIR = str(CONFIG_DIR)


def _dir_size(path: str) -> str:
    """Human-readable size of a directory tree, or '(not found)' if missing."""
    from pathlib import Path

    p = Path(path)
    if not p.exists():
        return "(not found)"
    total = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
    for unit in ("B", "KB", "MB", "GB"):
        if total < 1024:
            return f"{total:.1f} {unit}"
        total /= 1024
    return f"{total:.1f} TB"


@click.group(invoke_without_command=True)
@click.option("--device", default=None, help="Audio input device name or index.")
@click.option("--no-summarize", is_flag=True, help="Skip LLM summarization.")
@click.option("--diarize", is_flag=True, help="Enable speaker diarization (needs HF token).")
@click.option("--format", "output_format", type=click.Choice(["markdown", "json"]), default=None, help="Output format.")
@click.option("--model", default=None, help="Whisper model size (tiny, base, small, medium, large-v3).")
@click.option("--mic", is_flag=True, help="Also capture microphone input (mixed with system audio).")
@click.option("--mic-device", default=None, help="Specific mic device name (implies --mic).")
@click.option(
    "--keep-recording/--no-keep-recording",
    default=None,
    help="Keep or delete WAV recordings after transcription.",
)
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
    keep_recording: bool | None,
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
    if keep_recording is not None:
        config.output.keep_recording = keep_recording

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


@cli.command()
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation prompts.")
@click.option("--all", "all_", is_flag=True, help="Remove everything (config + cache + output).")
@click.option("--config", "config_", is_flag=True, help="Remove config directory (~/.config/ownscribe/).")
@click.option("--cache", is_flag=True, help="Remove cached binary (~/.local/share/ownscribe/).")
@click.option("--output", is_flag=True, help="Remove output directory with recordings/transcripts.")
@click.pass_context
def cleanup(
    ctx: click.Context,
    yes: bool,
    all_: bool,
    config_: bool,
    cache: bool,
    output: bool,
) -> None:
    """Remove ownscribe data from disk (config, cache, recordings)."""
    cfg = ctx.obj["config"]
    output_dir = str(cfg.output.resolved_dir)

    targets: list[tuple[str, str]] = []

    if all_:
        targets = [
            ("Config", _CONFIG_DIR),
            ("Cache", _CACHE_DIR),
            ("Output", output_dir),
        ]
    elif config_ or cache or output:
        if config_:
            targets.append(("Config", _CONFIG_DIR))
        if cache:
            targets.append(("Cache", _CACHE_DIR))
        if output:
            targets.append(("Output", output_dir))
    else:
        # Interactive: prompt for each directory
        for label, path in [
            ("Config", _CONFIG_DIR),
            ("Cache", _CACHE_DIR),
            ("Output", output_dir),
        ]:
            size = _dir_size(path)
            if size == "(not found)":
                click.echo(f"  {label}: {path} — {size}, skipping")
                continue
            if yes or click.confirm(f"  Remove {label}: {path} ({size})?"):
                targets.append((label, path))
        if not targets:
            click.echo("Nothing to remove.")
        return _remove_targets(targets)

    if not yes:
        click.echo("The following directories will be removed:")
        for label, path in targets:
            click.echo(f"  {label}: {path} ({_dir_size(path)})")
        if not click.confirm("Proceed?"):
            click.echo("Aborted.")
            return

    _remove_targets(targets)


def _remove_targets(targets: list[tuple[str, str]]) -> None:
    """Delete the listed directories."""
    from pathlib import Path

    for label, path in targets:
        p = Path(path)
        if p.exists():
            shutil.rmtree(p)
            click.echo(f"  Removed {label}: {path}")
        else:
            click.echo(f"  {label}: {path} — not found, skipping")
