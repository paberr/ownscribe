"""Core Audio Taps recorder — wraps the Swift notetaker-audio helper."""

from __future__ import annotations

import shutil
import signal
import subprocess
import sys
from pathlib import Path

from notetaker.audio.base import AudioRecorder

# Look for binary relative to package, then in PATH
_BINARY_CANDIDATES = [
    Path(__file__).resolve().parents[3] / "bin" / "notetaker-audio",  # dev: repo root
    Path(sys.prefix) / "bin" / "notetaker-audio",
]


def _find_binary() -> Path | None:
    for candidate in _BINARY_CANDIDATES:
        if candidate.exists() and candidate.is_file():
            return candidate
    # Fall back to PATH
    found = shutil.which("notetaker-audio")
    return Path(found) if found else None


class CoreAudioRecorder(AudioRecorder):
    """Records system audio using the notetaker-audio Swift helper."""

    def __init__(self, pid: int | None = None) -> None:
        self._pid = pid
        self._process: subprocess.Popen | None = None
        self._binary = _find_binary()
        self._silence_warning: bool = False

    def is_available(self) -> bool:
        return self._binary is not None

    def start(self, output_path: Path) -> None:
        if not self._binary:
            raise RuntimeError(
                "notetaker-audio binary not found. Run: bash swift/build.sh"
            )

        cmd = [str(self._binary), "capture", "--output", str(output_path)]
        if self._pid is not None:
            cmd.extend(["--pid", str(self._pid)])

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
                import click
                click.echo(stderr_output.strip(), err=True)

    def list_apps(self) -> str:
        if not self._binary:
            return "notetaker-audio binary not found. Run: bash swift/build.sh"
        result = subprocess.run(
            [str(self._binary), "list-apps"],
            capture_output=True,
            text=True,
        )
        return result.stdout
