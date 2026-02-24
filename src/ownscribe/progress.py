"""Progress indicators for long-running pipeline phases."""

from __future__ import annotations

import itertools
import logging
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
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
_TQDM_RE = re.compile(
    r"(?:(?P<filename>[^:\r\n]+):\s*)?"
    r"(?P<percent>[\d.]+)%\|.*?\|\s*"
    r"(?P<done>[\d.]+)\s*(?P<done_unit>[kKMGTPE]?i?B)\s*/\s*"
    r"(?P<total>[\d.]+)\s*(?P<total_unit>[kKMGTPE]?i?B)"
)
_BYTES_RE = re.compile(
    r"(?P<done>[\d.]+)\s*(?P<done_unit>[kKMGTPE]?i?B)\s*/\s*"
    r"(?P<total>[\d.]+)\s*(?P<total_unit>[kKMGTPE]?i?B)"
)
_PERCENT_RE = re.compile(r"(?P<percent>[\d.]+)%")


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


@dataclass
class DownloadProgressEvent:
    """Best-effort parsed progress for model downloads/preparation."""

    filename: str | None = None
    percent: float | None = None
    bytes_done: int | None = None
    bytes_total: int | None = None


def _parse_size_to_bytes(value: str, unit: str) -> int:
    multipliers = {
        "B": 1,
        "KB": 1024,
        "MB": 1024**2,
        "GB": 1024**3,
        "TB": 1024**4,
        "PB": 1024**5,
        "KIB": 1024,
        "MIB": 1024**2,
        "GIB": 1024**3,
        "TIB": 1024**4,
        "PIB": 1024**5,
    }
    factor = multipliers.get(unit.strip().upper())
    if factor is None:
        raise ValueError(f"Unknown size unit: {unit}")
    return int(float(value) * factor)


def parse_download_progress(text: str) -> DownloadProgressEvent | None:
    """Parse a tqdm/HF-style progress line into a structured event."""
    clean = _ANSI_RE.sub("", text).strip()
    if not clean:
        return None

    if m := _TQDM_RE.search(clean):
        filename = (m.group("filename") or "").strip() or None
        return DownloadProgressEvent(
            filename=filename,
            percent=float(m.group("percent")),
            bytes_done=_parse_size_to_bytes(m.group("done"), m.group("done_unit")),
            bytes_total=_parse_size_to_bytes(m.group("total"), m.group("total_unit")),
        )

    if m := _BYTES_RE.search(clean):
        percent = None
        if m2 := _PERCENT_RE.search(clean):
            percent = float(m2.group("percent"))
        return DownloadProgressEvent(
            percent=percent,
            bytes_done=_parse_size_to_bytes(m.group("done"), m.group("done_unit")),
            bytes_total=_parse_size_to_bytes(m.group("total"), m.group("total_unit")),
        )

    if m := _PERCENT_RE.search(clean):
        return DownloadProgressEvent(percent=float(m.group("percent")))

    return None


def _human_bytes(value: int) -> str:
    size = float(value)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            if unit == "B":
                return f"{int(size)} {unit}"
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def format_download_progress(event: DownloadProgressEvent, *, include_percent: bool = True) -> str:
    """Format a parsed download progress event for display in the TUI."""
    parts: list[str] = []
    if event.filename:
        parts.append(event.filename)
    if event.bytes_done is not None and event.bytes_total is not None:
        parts.append(f"{_human_bytes(event.bytes_done)} / {_human_bytes(event.bytes_total)}")
    if include_percent and event.percent is not None:
        parts.append(f"{int(event.percent)}%")
    return " ".join(parts).strip()


def download_event_fraction(event: DownloadProgressEvent) -> float | None:
    """Convert a parsed download event to a progress-bar fraction."""
    if event.bytes_done is not None and event.bytes_total and event.bytes_total > 0:
        return max(0.0, min(1.0, event.bytes_done / event.bytes_total))
    if event.percent is not None:
        return max(0.0, min(1.0, event.percent / 100.0))
    return None


class DownloadProgressWriter:
    """File-like object that parses download progress from captured output."""

    def __init__(self, update_fn: Callable[[DownloadProgressEvent], None]) -> None:
        self._update_fn = update_fn
        self._buffer = ""

    def write(self, text: str) -> int:
        self._buffer += text
        while True:
            idx_r = self._buffer.find("\r")
            idx_n = self._buffer.find("\n")
            idxs = [idx for idx in (idx_r, idx_n) if idx != -1]
            if not idxs:
                break
            idx = min(idxs)
            chunk = self._buffer[:idx]
            self._buffer = self._buffer[idx + 1:]
            self._consume(chunk)
        return len(text)

    def flush(self) -> None:
        if self._buffer:
            self._consume(self._buffer)
            self._buffer = ""

    def _consume(self, chunk: str) -> None:
        try:
            event = parse_download_progress(chunk)
        except (ValueError, OverflowError):
            logging.getLogger(__name__).debug("Ignoring malformed download progress output: %r", chunk, exc_info=True)
            return
        if event:
            self._update_fn(event)


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

    def __init__(
        self,
        *,
        diarize: bool = False,
        summarize: bool = False,
        transcribe: bool = True,
        include_prepare: bool = False,
        download_summarizer: bool = False,
    ) -> None:
        steps: list[_Step] = []
        if include_prepare:
            steps.append(_Step("preparing_models", "Preparing models", indent=0))
        if transcribe:
            steps.append(_Step("transcribing", "Transcribing", indent=0))
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
            if download_summarizer:
                steps.append(_Step("downloading_model", "Downloading model", indent=1))
        elif download_summarizer:
            steps.append(_Step("downloading_model", "Downloading summarization model", indent=0))
        self._steps = steps
        self._step_map: dict[str, _Step] = {s.key: s for s in steps}
        self._active: set[str] = set()
        self._completed: set[str] = set()
        self._progress: dict[str, float] = {}
        self._details: dict[str, str] = {}
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
                self._details.pop(key, None)
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
                    self._progress.pop(other_key, None)
                    self._details.pop(other_key, None)
            self._active.add(key)
            self._progress.pop(key, None)
            self._details.pop(key, None)
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
            self._details.pop(key, None)
            # If top-level step, also complete any active sub-steps
            if step.indent == 0:
                for s in self._steps:
                    if s.indent > 0 and s.key in self._active:
                        self._active.discard(s.key)
                        self._completed.add(s.key)
                        self._progress.pop(s.key, None)
                        self._details.pop(s.key, None)

    def fail(self, key: str) -> None:
        """Mark a step as failed — removes from active without completing."""
        with self._lock:
            self._active.discard(key)
            self._progress.pop(key, None)
            self._details.pop(key, None)

    def update(self, key: str, fraction: float) -> None:
        with self._lock:
            if key in self._step_map:
                self._progress[key] = max(0.0, min(1.0, fraction))

    def set_detail(self, key: str, text: str | None) -> None:
        with self._lock:
            if key not in self._step_map:
                return
            if text:
                self._details[key] = text
            else:
                self._details.pop(key, None)

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
            details = dict(self._details)

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
                    lines.append(f"{indent}{frame} {step.label:<20s} [{bar}] {pct:3d}%")
                else:
                    lines.append(f"{indent}{frame} {step.label}")
                if detail := details.get(step.key):
                    lines.append(f"{indent}  {detail}")
            else:
                lines.append(f"{indent}\u25cb {step.label}")

        # Move cursor up to overwrite previous render
        prev_lines = self._lines_rendered
        if prev_lines > 0:
            self._stderr.write(f"\033[{prev_lines}A")

        # If the render shrinks (e.g. detail line disappears), explicitly clear the
        # now-stale trailing rows by writing blank cleared lines.
        render_lines = list(lines)
        if prev_lines > len(render_lines):
            render_lines.extend([""] * (prev_lines - len(render_lines)))

        for line in render_lines:
            self._stderr.write(f"{line}\033[K\n")
        self._stderr.flush()
        self._lines_rendered = len(render_lines)

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

    def set_detail(self, key: str, text: str | None) -> None:
        pass

    def diarization_hook(self, step_name: str, _artifact, **kwargs) -> None:
        pass
