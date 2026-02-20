"""Progress indicators for long-running pipeline phases."""

from __future__ import annotations

import itertools
import re
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

_BRAILLE = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
_FILLED = "█"
_EMPTY = "░"
_BAR_WIDTH = 20
_INTERVAL = 0.1

_PROGRESS_RE = re.compile(r"Progress:\s*([\d.]+)%")


class Spinner:
    """Indeterminate spinner shown during phases with unknown duration."""

    def __init__(self, label: str) -> None:
        self._label = label
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._stderr = sys.stderr

    def __enter__(self) -> Spinner:
        self._stderr = sys.stderr
        self._stop.clear()
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_exc) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join()
        self._stderr.write(f"\r  \u2714 {self._label} done.\033[K\n")
        self._stderr.flush()

    def update(self, label: str) -> None:
        self._label = label

    def _spin(self) -> None:
        for frame in itertools.cycle(_BRAILLE):
            if self._stop.is_set():
                break
            self._stderr.write(f"\r  {frame} {self._label}\033[K")
            self._stderr.flush()
            time.sleep(_INTERVAL)


class ProgressBar:
    """Determinate progress bar for phases with known percentage."""

    def __init__(self, label: str) -> None:
        self._label = label
        self._fraction = 0.0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._stderr = sys.stderr

    def __enter__(self) -> ProgressBar:
        self._stderr = sys.stderr
        self._stop.clear()
        self._thread = threading.Thread(target=self._animate, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_exc) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join()
        self._stderr.write(f"\r  \u2714 {self._label} done.\033[K\n")
        self._stderr.flush()

    def update(self, fraction: float) -> None:
        self._fraction = max(0.0, min(1.0, fraction))

    def _render(self, spinner_char: str) -> str:
        filled = int(self._fraction * _BAR_WIDTH)
        bar = _FILLED * filled + _EMPTY * (_BAR_WIDTH - filled)
        pct = int(self._fraction * 100)
        return f"\r  {self._label:<20s} [{bar}] {pct:3d}%\033[K"

    def _animate(self) -> None:
        for frame in itertools.cycle(_BRAILLE):
            if self._stop.is_set():
                break
            self._stderr.write(self._render(frame))
            self._stderr.flush()
            time.sleep(_INTERVAL)


class ProgressWriter:
    """File-like object that parses WhisperX progress output and feeds an update callable."""

    def __init__(self, update_fn: Callable[[float], None], offset: float = 0.0, scale: float = 1.0) -> None:
        self._update_fn = update_fn
        self._offset = offset
        self._scale = scale

    def write(self, text: str) -> int:
        m = _PROGRESS_RE.search(text)
        if m:
            raw = float(m.group(1)) / 100.0
            self._update_fn(self._offset + raw * self._scale)
        return len(text)

    def flush(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Pipeline-level checklist progress
# ---------------------------------------------------------------------------

_DIAR_KEY_MAP = {"discrete_diarization": "clustering"}

# Ordered sub-step keys for diarization, used to auto-complete the previous
# sub-step when a new one starts.
_DIAR_SUBSTEPS = ("segmentation", "speaker_counting", "embeddings", "clustering")


@dataclass
class _Step:
    key: str
    label: str
    indent: int = 0


class PipelineProgress:
    """Full-pipeline checklist display."""

    def __init__(self, *, diarize: bool = False, summarize: bool = False) -> None:
        steps: list[_Step] = [_Step("transcribing", "Transcribing", indent=0)]
        if diarize:
            steps.append(_Step("diarizing", "Diarizing", indent=0))
            steps.extend([
                _Step("segmentation", "Segmentation", indent=1),
                _Step("speaker_counting", "Speaker counting", indent=1),
                _Step("embeddings", "Embeddings", indent=1),
                _Step("clustering", "Clustering", indent=1),
            ])
        if summarize:
            steps.append(_Step("summarizing", "Summarizing", indent=0))
        self._steps = steps
        self._step_map: dict[str, _Step] = {s.key: s for s in steps}
        self._active: set[str] = set()
        self._completed: set[str] = set()
        self._progress: dict[str, float] = {}
        self._lock = threading.Lock()
        self._lines_rendered = 0
        self._stderr = sys.stderr
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # -- context manager -----------------------------------------------------

    def __enter__(self) -> PipelineProgress:
        self._stderr = sys.stderr
        return self

    def __exit__(self, *_exc) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        # Final render: mark any remaining active steps as completed
        with self._lock:
            for key in list(self._active):
                self._completed.add(key)
                self._progress.pop(key, None)
            self._active.clear()
        self._render_all(final=True)

    # -- public API ----------------------------------------------------------

    def begin(self, key: str) -> None:
        with self._lock:
            step = self._step_map.get(key)
            if step is None:
                return
            # Auto-complete any active sibling at the same indent level
            for other_key in list(self._active):
                other = self._step_map[other_key]
                if other.indent == step.indent:
                    self._active.discard(other_key)
                    self._completed.add(other_key)
            self._active.add(key)
            self._progress.pop(key, None)
        # Lazy-start animation thread on first begin()
        if self._thread is None:
            self._stop.clear()
            self._thread = threading.Thread(target=self._animate, daemon=True)
            self._thread.start()

    def complete(self, key: str) -> None:
        with self._lock:
            step = self._step_map.get(key)
            if step is None:
                return
            self._active.discard(key)
            self._completed.add(key)
            self._progress.pop(key, None)
            # If top-level step, also complete any active sub-steps
            if step.indent == 0:
                for s in self._steps:
                    if s.indent > 0 and s.key in self._active:
                        self._active.discard(s.key)
                        self._completed.add(s.key)
                        self._progress.pop(s.key, None)

    def fail(self, key: str) -> None:
        """Mark a step as failed — removes from active without completing."""
        with self._lock:
            self._active.discard(key)
            self._progress.pop(key, None)

    def update(self, key: str, fraction: float) -> None:
        with self._lock:
            if key in self._step_map:
                self._progress[key] = max(0.0, min(1.0, fraction))

    def diarization_hook(self, step_name: str, _artifact, **kwargs) -> None:
        """Pyannote-compatible hook callback for diarization progress."""
        # Map pyannote step names to our keys
        short = step_name.rsplit("/", 1)[-1] if "/" in step_name else step_name
        key = _DIAR_KEY_MAP.get(short, short)

        if key not in self._step_map:
            return

        completed = kwargs.get("completed")
        total = kwargs.get("total")

        # Auto-transition: complete previous sub-steps and begin new one
        if key not in self._active and key not in self._completed:
            self.begin(key)

        if completed is not None and total:
            self.update(key, completed / total)

    # -- rendering -----------------------------------------------------------

    def _render_all(self, *, final: bool = False) -> None:
        with self._lock:
            active = set(self._active)
            completed = set(self._completed)
            progress = dict(self._progress)

        # Pick a spinner frame (not needed for final)
        frame = ""
        if not final:
            frame = _BRAILLE[int(time.time() / _INTERVAL) % len(_BRAILLE)]

        lines: list[str] = []
        for step in self._steps:
            indent = "    " if step.indent == 1 else "  "

            if step.key in completed:
                lines.append(f"{indent}\u2714 {step.label} done.")
            elif step.key in active:
                frac = progress.get(step.key)
                if frac is not None:
                    filled = int(frac * _BAR_WIDTH)
                    bar = _FILLED * filled + _EMPTY * (_BAR_WIDTH - filled)
                    pct = int(frac * 100)
                    lines.append(f"{indent}{step.label:<20s} [{bar}] {pct:3d}%")
                else:
                    lines.append(f"{indent}{frame} {step.label}")
            else:
                lines.append(f"{indent}\u25cb {step.label}")

        # Move cursor up to overwrite previous render
        if self._lines_rendered > 0:
            self._stderr.write(f"\033[{self._lines_rendered}A")

        output = "\n".join(lines)
        self._stderr.write(f"{output}\033[K\n")
        self._stderr.flush()
        self._lines_rendered = len(lines)

    def _animate(self) -> None:
        while not self._stop.is_set():
            self._render_all()
            time.sleep(_INTERVAL)


class NullProgress:
    """No-op progress for when no display is needed."""

    def begin(self, key: str) -> None:
        pass

    def complete(self, key: str) -> None:
        pass

    def fail(self, key: str) -> None:
        pass

    def update(self, key: str, fraction: float) -> None:
        pass

    def diarization_hook(self, step_name: str, _artifact, **kwargs) -> None:
        pass
