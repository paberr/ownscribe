"""Fallback recorder using sounddevice (mic or virtual device)."""

from __future__ import annotations

import threading
from pathlib import Path

import sounddevice as sd
import soundfile as sf

from ownscribe.audio.base import AudioRecorder


class SoundDeviceRecorder(AudioRecorder):
    """Records from any audio input device using sounddevice + soundfile."""

    def __init__(self, device: str | int | None = None, samplerate: int = 48000, channels: int = 1) -> None:
        self._device = device
        self._samplerate = samplerate
        self._channels = channels
        self._stream: sd.InputStream | None = None
        self._file: sf.SoundFile | None = None
        self._lock = threading.Lock()

    def is_available(self) -> bool:
        try:
            sd.query_devices()
            return True
        except Exception:
            return False

    def start(self, output_path: Path) -> None:
        self._file = sf.SoundFile(
            str(output_path),
            mode="w",
            samplerate=self._samplerate,
            channels=self._channels,
            format="WAV",
            subtype="FLOAT",
        )

        def callback(indata, frames, time, status):
            with self._lock:
                if self._file is not None:
                    self._file.write(indata.copy())

        self._stream = sd.InputStream(
            device=self._device,
            samplerate=self._samplerate,
            channels=self._channels,
            callback=callback,
        )
        self._stream.start()

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        with self._lock:
            if self._file is not None:
                self._file.close()
                self._file = None

    @staticmethod
    def list_devices() -> str:
        return str(sd.query_devices())
