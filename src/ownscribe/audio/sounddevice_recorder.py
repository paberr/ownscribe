"""Fallback recorder using sounddevice (mic or virtual device)."""

from __future__ import annotations

import threading
import time as _time
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf

from ownscribe.audio.base import AudioRecorder

_SILENCE_THRESHOLD = 1e-4  # ~-80 dB


class SoundDeviceRecorder(AudioRecorder):
    """Records from any audio input device using sounddevice + soundfile."""

    def __init__(
        self,
        device: str | int | None = None,
        samplerate: int = 48000,
        channels: int = 1,
        silence_timeout: int = 0,
    ) -> None:
        self._device = device
        self._samplerate = samplerate
        self._channels = channels
        self._silence_timeout = silence_timeout
        self._stream: sd.InputStream | None = None
        self._file: sf.SoundFile | None = None
        self._lock = threading.Lock()
        self._last_loud_time: float = 0.0
        self._timed_out: bool = False

    def is_available(self) -> bool:
        try:
            sd.query_devices()
            return True
        except Exception:
            return False

    def start(self, output_path: Path) -> None:
        self._last_loud_time = _time.monotonic()
        self._timed_out = False

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

            # Silence tracking
            if self._silence_timeout > 0:
                peak = np.max(np.abs(indata))
                if peak > _SILENCE_THRESHOLD:
                    self._last_loud_time = _time.monotonic()
                elif _time.monotonic() - self._last_loud_time > self._silence_timeout:
                    self._timed_out = True
                    raise sd.CallbackStop

        self._stream = sd.InputStream(
            device=self._device,
            samplerate=self._samplerate,
            channels=self._channels,
            callback=callback,
        )
        self._stream.start()

    @property
    def is_recording(self) -> bool:
        return (
            not self._timed_out
            and self._stream is not None
            and getattr(self._stream, "active", False)
        )

    @property
    def silence_timed_out(self) -> bool:
        return self._timed_out

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
