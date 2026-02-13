"""Core Audio Taps recorder — wraps the Swift ownscribe-audio helper."""

from __future__ import annotations

import platform
import shutil
import signal
import subprocess
import sys
import urllib.request
from pathlib import Path

import click

from ownscribe.audio.base import AudioRecorder

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

    def __init__(self, mic: bool = False, mic_device: str = "") -> None:
        self._mic = mic
        self._mic_device = mic_device
        self._process: subprocess.Popen | None = None
        self._binary = _find_binary()
        self._silence_warning: bool = False

    def is_available(self) -> bool:
        return self._binary is not None

    def start(self, output_path: Path) -> None:
        if not self._binary:
            raise RuntimeError("ownscribe-audio binary not found. Run: bash swift/build.sh")

        cmd = [str(self._binary), "capture", "--output", str(output_path)]
        if self._mic or self._mic_device:
            cmd.append("--mic")
        if self._mic_device:
            cmd.extend(["--mic-device", self._mic_device])

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

    def stop(self) -> None:
        if self._process and self._process.poll() is None:
            self._process.send_signal(signal.SIGINT)
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.terminate()
                self._process.wait(timeout=5)
        if self._process and self._process.stderr:
            stderr_output = self._process.stderr.read().decode(errors="replace")
            if stderr_output:
                if "[SILENCE_WARNING]" in stderr_output:
                    self._silence_warning = True
                click.echo(stderr_output.strip(), err=True)

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
