"""Core Audio Taps recorder — wraps the Swift ownscribe-audio helper."""

from __future__ import annotations

import logging
import platform
import shutil
import signal
import subprocess
import sys
import urllib.request
from pathlib import Path

import click

from ownscribe.audio.base import AudioRecorder

logger = logging.getLogger(__name__)

_WAV_HEADER_SIZE = 44

# Stderr markers emitted by the Swift helper (keep in sync with AudioCapture.swift)
_MARKER_SILENCE_WARNING = "[SILENCE_WARNING]"
_MARKER_SILENCE_TIMEOUT = "[SILENCE_TIMEOUT]"
_MARKER_STREAM_ERROR = "[STREAM_ERROR]"
_MARKER_STREAM_DIED = "[STREAM_DIED]"

# Timeouts for graceful shutdown of the Swift helper.
# SIGINT triggers track merging (system + mic) which can take a while for long recordings.
# wait() returns immediately on exit, so generous values cost nothing in the normal case.
_STOP_TIMEOUT = 30  # seconds to wait after SIGINT (allows track merging)
_KILL_TIMEOUT = 10  # seconds to wait after SIGTERM before SIGKILL

# Look for binary relative to package, then in PATH
_BINARY_CANDIDATES = [
    Path(__file__).resolve().parents[3] / "bin" / "ownscribe-audio",  # dev: repo root
    Path(sys.prefix) / "bin" / "ownscribe-audio",
]

_CACHE_DIR = Path.home() / ".local" / "share" / "ownscribe" / "bin"
_DOWNLOAD_URL = "https://github.com/paberr/ownscribe/releases/latest/download/ownscribe-audio-{arch}"


def _download_binary() -> Path | None:
    """Download the prebuilt ownscribe-audio binary from GitHub Releases."""
    if sys.platform != "darwin":
        return None

    arch = platform.machine()
    if arch not in ("arm64", "x86_64"):
        return None

    url = _DOWNLOAD_URL.format(arch=arch)
    dest = _CACHE_DIR / "ownscribe-audio"

    try:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        click.echo(f"Downloading ownscribe-audio ({arch}) from GitHub Releases...")
        urllib.request.urlretrieve(url, dest)
        dest.chmod(0o755)
        click.echo(f"Saved to {dest}")
        return dest
    except Exception as e:
        click.echo(f"Download failed: {e}", err=True)
        # Clean up partial download
        dest.unlink(missing_ok=True)
        return None


def _find_binary() -> Path | None:
    for candidate in _BINARY_CANDIDATES:
        if candidate.exists() and candidate.is_file():
            return candidate

    # Check cache directory
    cached = _CACHE_DIR / "ownscribe-audio"
    if cached.exists() and cached.is_file():
        return cached

    # Fall back to PATH
    found = shutil.which("ownscribe-audio")
    if found:
        return Path(found)

    # Try downloading
    return _download_binary()


class CoreAudioRecorder(AudioRecorder):
    """Records system audio using the ownscribe-audio Swift helper."""

    def __init__(self, mic: bool = False, mic_device: str = "", silence_timeout: int = 0) -> None:
        self._mic = mic
        self._mic_device = mic_device
        self._silence_timeout = silence_timeout
        self._process: subprocess.Popen | None = None
        self._binary = _find_binary()
        self._silence_warning: bool = False
        self._silence_timed_out: bool = False
        self._muted: bool = False
        self._output_path: Path | None = None
        self._crashed: bool = False
        self._exit_code: int | None = None
        self._stderr_output: str = ""

    def is_available(self) -> bool:
        return self._binary is not None

    def start(self, output_path: Path) -> None:
        if not self._binary:
            raise RuntimeError("ownscribe-audio binary not found. Run: bash swift/build.sh")

        self._output_path = output_path
        cmd = [str(self._binary), "capture", "--output", str(output_path)]
        if self._mic or self._mic_device:
            cmd.append("--mic")
        if self._mic_device:
            cmd.extend(["--mic-device", self._mic_device])
        if self._silence_timeout > 0:
            cmd.extend(["--silence-timeout", str(self._silence_timeout)])

        self._process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            process_group=0,
        )

    @property
    def silence_warning(self) -> bool:
        """True if the Swift helper reported a silence warning."""
        return self._silence_warning

    def toggle_mute(self) -> None:
        """Send SIGUSR1 to the Swift helper to toggle mic mute."""
        if self._mic and self._process and self._process.poll() is None:
            self._process.send_signal(signal.SIGUSR1)
            self._muted = not self._muted

    @property
    def is_muted(self) -> bool:
        return self._muted

    @property
    def is_recording(self) -> bool:
        return self._process is not None and self._process.poll() is None

    @property
    def silence_timed_out(self) -> bool:
        return self._silence_timed_out

    @property
    def crashed(self) -> bool:
        return self._crashed

    @property
    def exit_code(self) -> int | None:
        return self._exit_code

    @property
    def stderr_output(self) -> str:
        return self._stderr_output

    def stop(self) -> None:
        rc = self._process.poll() if self._process else None

        if rc is None and self._process:
            self._process.send_signal(signal.SIGINT)
            try:
                self._process.wait(timeout=_STOP_TIMEOUT)
            except subprocess.TimeoutExpired:
                self._process.terminate()
                try:
                    self._process.wait(timeout=_KILL_TIMEOUT)
                except subprocess.TimeoutExpired:
                    self._process.kill()
                    self._process.wait()

        if self._process:
            self._exit_code = self._process.returncode
            self._finalize_stderr()

        if self._crashed:
            self._attempt_recovery()

    def _finalize_stderr(self) -> None:
        if not self._process or not self._process.stderr:
            return
        self._stderr_output = self._process.stderr.read().decode(errors="replace")
        if not self._stderr_output:
            return

        if _MARKER_SILENCE_WARNING in self._stderr_output:
            self._silence_warning = True
        if _MARKER_SILENCE_TIMEOUT in self._stderr_output:
            self._silence_timed_out = True
        if _MARKER_STREAM_ERROR in self._stderr_output or _MARKER_STREAM_DIED in self._stderr_output:
            self._crashed = True
        if self._exit_code is not None and self._exit_code != 0:
            self._crashed = True

        _NOISE_PREFIXES = ("Recording ", "Saved ", "Merged audio saved")
        _NOISE_LINES = (_MARKER_SILENCE_TIMEOUT, "[MIC_MUTED]", "[MIC_UNMUTED]")
        lines = [
            line for line in self._stderr_output.strip().splitlines()
            if line not in _NOISE_LINES
            and not line.startswith(_NOISE_PREFIXES)
        ]
        if lines:
            click.echo("\n".join(lines), err=True)

    def _attempt_recovery(self) -> None:
        if not self._output_path:
            return
        output = self._output_path

        if output.exists() and output.stat().st_size > _WAV_HEADER_SIZE:
            return

        sys_tmp = Path(str(output) + ".sys.tmp.wav")
        mic_tmp = Path(str(output) + ".mic.tmp.wav")
        has_sys = sys_tmp.exists() and sys_tmp.stat().st_size > _WAV_HEADER_SIZE
        has_mic = mic_tmp.exists() and mic_tmp.stat().st_size > _WAV_HEADER_SIZE

        if not has_sys and not has_mic:
            return
        if not shutil.which("ffmpeg"):
            logger.warning("ffmpeg not found — cannot recover audio from tmp files")
            return

        click.echo("\n  Recovering audio from temp files...", err=True)
        try:
            if has_sys and has_mic:
                subprocess.run(
                    [
                        "ffmpeg", "-y", "-loglevel", "error",
                        "-i", str(mic_tmp), "-i", str(sys_tmp),
                        "-filter_complex",
                        "[0:a]aresample=24000[mic];[mic][1:a]amix=inputs=2:duration=longest[out]",
                        "-map", "[out]", "-ar", "24000", "-ac", "1",
                        "-c:a", "pcm_f32le", str(output),
                    ],
                    check=True, capture_output=True,
                )
            elif has_sys:
                sys_tmp.rename(output)
            elif has_mic:
                subprocess.run(
                    [
                        "ffmpeg", "-y", "-loglevel", "error",
                        "-i", str(mic_tmp), "-ar", "24000", "-ac", "1",
                        "-c:a", "pcm_f32le", str(output),
                    ],
                    check=True, capture_output=True,
                )

            if output.exists() and output.stat().st_size > _WAV_HEADER_SIZE:
                sys_tmp.unlink(missing_ok=True)
                mic_tmp.unlink(missing_ok=True)
                click.echo("  Audio recovered successfully.", err=True)
            else:
                click.echo("  Recovery produced empty file.", err=True)
        except subprocess.CalledProcessError as e:
            stderr = e.stderr.decode(errors="replace") if e.stderr else ""
            logger.warning("ffmpeg recovery failed: %s", stderr)
            click.echo(f"  Recovery failed: {stderr}", err=True)

    def list_devices(self) -> str:
        """List available audio devices using the Swift helper."""
        if not self._binary:
            return "ownscribe-audio binary not found. Run: bash swift/build.sh"
        result = subprocess.run(
            [str(self._binary), "list-devices"],
            capture_output=True,
            text=True,
        )
        return result.stdout

    def list_apps(self) -> str:
        if not self._binary:
            return "ownscribe-audio binary not found. Run: bash swift/build.sh"
        result = subprocess.run(
            [str(self._binary), "list-apps"],
            capture_output=True,
            text=True,
        )
        return result.stdout
