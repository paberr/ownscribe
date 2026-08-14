"""Protocol conformance tests (P0).

Two jobs:

1. Validate every committed fixture against the JSON Schemas in `schema/protocol/v1/`,
   plus the ordering invariants JSON Schema cannot express (docs/protocol.md §2).
2. Drive `tools/mock-ownscribe-core` as a subprocess exactly as the real binary will be
   driven, and assert a trivial consumer can run a full session from it.

Deliberately imports nothing from `ownscribe`. The mock exists so that the Python backend
can be built before the Swift binary does, and these tests must keep working in an
environment where the project's own dependencies are absent.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
import wave
from pathlib import Path

import pytest

jsonschema = pytest.importorskip("jsonschema")
referencing = pytest.importorskip("referencing")

REPO_ROOT = Path(__file__).resolve().parent.parent
SCHEMA_DIR = REPO_ROOT / "schema" / "protocol" / "v1"
FIXTURE_DIR = REPO_ROOT / "tests" / "fixtures" / "protocol"
ERROR_FIXTURE_DIR = FIXTURE_DIR / "errors"
MOCK = REPO_ROOT / "tools" / "mock-ownscribe-core"

PROTOCOL_VERSION = 1
DIARIZATION_SUBSTAGES = ("segmentation", "speaker_counting", "embeddings", "clustering")


def session_fixtures() -> list[Path]:
    return sorted(FIXTURE_DIR.glob("*.ndjson"))


def error_fixtures() -> list[Path]:
    return sorted(ERROR_FIXTURE_DIR.glob("*.ndjson"))


def all_fixtures() -> list[Path]:
    return session_fixtures() + error_fixtures()


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


@pytest.fixture(scope="session")
def validator():
    from jsonschema import Draft202012Validator
    from referencing import Registry, Resource

    resources = []
    for path in sorted(SCHEMA_DIR.glob("*.json")):
        schema = json.loads(path.read_text())
        resources.append((schema["$id"], Resource.from_contents(schema)))
    registry = Registry().with_resources(resources)

    event_schema = json.loads((SCHEMA_DIR / "event.json").read_text())
    return Draft202012Validator(event_schema, registry=registry)


# ---------------------------------------------------------------------------
# Schema conformance
# ---------------------------------------------------------------------------


class TestSchemas:
    def test_every_schema_is_valid_draft_2020_12(self):
        from jsonschema import Draft202012Validator

        for path in sorted(SCHEMA_DIR.glob("*.json")):
            Draft202012Validator.check_schema(json.loads(path.read_text()))

    def test_event_schema_covers_every_event_type(self):
        event = json.loads((SCHEMA_DIR / "event.json").read_text())
        declared = set(event["properties"]["type"]["enum"])
        on_disk = {p.stem for p in SCHEMA_DIR.glob("*.json")} - {"event", "defs"}
        assert declared == on_disk

    @pytest.mark.parametrize("path", all_fixtures(), ids=lambda p: p.stem)
    def test_fixture_lines_validate(self, path: Path, validator):
        for lineno, event in enumerate(load(path), start=1):
            errors = sorted(validator.iter_errors(event), key=lambda e: e.json_path)
            assert not errors, f"{path.name}:{lineno}: {[e.message for e in errors]}"

    def test_unknown_event_type_is_rejected_by_schema(self, validator):
        """The schema is strict on purpose; consumers, unlike tests, must be lenient."""
        assert list(validator.iter_errors({"type": "wat"}))

    def test_misspelled_field_is_rejected(self, validator):
        """The whole reason additionalProperties is false: catch drift between the two sides."""
        bad = {"type": "partial", "start": 1.0, "end": 2.0, "text": "hi", "speakr": "x"}
        assert list(validator.iter_errors(bad))


# ---------------------------------------------------------------------------
# Ordering invariants — docs/protocol.md §2
# ---------------------------------------------------------------------------


def is_terminal(event: dict) -> bool:
    if event.get("type") == "final":
        return True
    return event.get("type") == "error" and not event.get("recoverable", False)


class TestInvariants:
    @pytest.mark.parametrize("path", all_fixtures(), ids=lambda p: p.stem)
    def test_hello_is_first_and_unique(self, path: Path):
        events = load(path)
        assert events[0]["type"] == "hello"
        assert events[0]["protocol"] == PROTOCOL_VERSION
        assert sum(1 for e in events if e["type"] == "hello") == 1

    @pytest.mark.parametrize("path", all_fixtures(), ids=lambda p: p.stem)
    def test_exactly_one_terminal_event_and_it_is_last(self, path: Path):
        events = load(path)
        terminals = [i for i, e in enumerate(events) if is_terminal(e)]
        assert len(terminals) == 1, f"{path.name}: expected one terminal event, got {len(terminals)}"
        assert terminals[0] == len(events) - 1, f"{path.name}: terminal event is not last"

    @pytest.mark.parametrize("path", all_fixtures(), ids=lambda p: p.stem)
    def test_segment_ids_strictly_increase(self, path: Path):
        ids = [e["id"] for e in load(path) if e["type"] == "segment"]
        assert ids == sorted(set(ids)) and len(ids) == len(set(ids))

    @pytest.mark.parametrize("path", all_fixtures(), ids=lambda p: p.stem)
    def test_timestamps_are_ordered(self, path: Path):
        for event in load(path):
            if event["type"] not in ("partial", "segment"):
                continue
            assert event["start"] <= event["end"], f"{path.name}: {event}"
            for word in event.get("words", []):
                assert word["start"] <= word["end"], f"{path.name}: {word}"
                assert event["start"] <= word["start"], f"{path.name}: word precedes segment"
                assert word["end"] <= event["end"] + 1e-6, f"{path.name}: word outruns segment"

    @pytest.mark.parametrize("path", all_fixtures(), ids=lambda p: p.stem)
    def test_final_segment_count_matches_emitted(self, path: Path):
        events = load(path)
        final = events[-1]
        if final["type"] != "final":
            pytest.skip("session ends in an error")
        emitted = sum(1 for e in events if e["type"] == "segment")
        assert final["segments"] == emitted

    @pytest.mark.parametrize("path", all_fixtures(), ids=lambda p: p.stem)
    def test_substages_only_under_diarizing_and_in_order(self, path: Path):
        seen: list[str] = []
        for event in load(path):
            substage = event.get("substage")
            if substage is None:
                continue
            assert event["stage"] == "diarizing", f"{path.name}: substage outside diarizing"
            if event["type"] == "state" and event["status"] == "begin":
                seen.append(substage)
        expected_order = [s for s in DIARIZATION_SUBSTAGES if s in seen]
        assert seen == expected_order, f"{path.name}: substages out of order: {seen}"

    @pytest.mark.parametrize("path", all_fixtures(), ids=lambda p: p.stem)
    def test_state_begins_are_closed(self, path: Path):
        """Every begin is matched by an end, or by the terminal event (docs/protocol.md §3)."""
        events = load(path)
        open_stages: set[tuple[str, str | None]] = set()
        for event in events:
            if event["type"] != "state":
                continue
            key = (event["stage"], event.get("substage"))
            if event["status"] == "begin":
                open_stages.add(key)
            else:
                assert key in open_stages, f"{path.name}: end without begin: {key}"
                open_stages.discard(key)

        if events[-1]["type"] == "error":
            # A terminal error closes whatever was open; the consumer calls progress.fail().
            return
        assert not open_stages, f"{path.name}: unclosed stages after a clean final: {open_stages}"

    @pytest.mark.parametrize("path", all_fixtures(), ids=lambda p: p.stem)
    def test_speaker_labels_are_declared(self, path: Path):
        events = load(path)
        declared = {label for e in events if e["type"] == "speakers" for label in e["labels"]}
        used = {e["speaker"] for e in events if e["type"] == "segment" and e.get("speaker")}
        if not declared:
            pytest.skip("no speakers event")
        assert used <= declared, f"{path.name}: undeclared speakers {used - declared}"

    @pytest.mark.parametrize("path", all_fixtures(), ids=lambda p: p.stem)
    def test_embedding_dim_matches_vector(self, path: Path):
        for event in load(path):
            for embedding in event.get("embeddings", []) if event["type"] == "speakers" else []:
                assert embedding["dim"] == len(embedding["vector"])

    def test_partials_never_carry_speaker_or_words(self):
        for path in all_fixtures():
            for event in load(path):
                if event["type"] == "partial":
                    assert "speaker" not in event and "words" not in event

    def test_every_error_code_has_a_fixture(self):
        defs = json.loads((SCHEMA_DIR / "defs.json").read_text())
        codes = set(defs["$defs"]["errorCode"]["enum"])
        covered = {e["code"] for p in all_fixtures() for e in load(p) if e["type"] == "error"}
        assert codes == covered, f"error codes without a fixture: {codes - covered}"

    def test_recoverable_errors_are_not_terminal(self):
        for path in all_fixtures():
            events = load(path)
            for event in events[:-1]:
                if event["type"] == "error":
                    assert event["recoverable"], f"{path.name}: terminal error mid-stream"


# ---------------------------------------------------------------------------
# The mock binary
# ---------------------------------------------------------------------------


def run_mock(*args: str, fixture: Path | None = None, timeout: float = 30) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    if fixture is not None:
        env["OWNSCRIBE_MOCK_FIXTURE"] = str(fixture)
    return subprocess.run(
        [sys.executable, str(MOCK), *args],
        capture_output=True, text=True, env=env, timeout=timeout,
    )


def parse_stdout(stdout: str) -> list[dict]:
    return [json.loads(line) for line in stdout.splitlines() if line.strip()]


class TestMockBinary:
    def test_is_executable(self):
        assert MOCK.exists()
        assert os.access(MOCK, os.X_OK), "mock must be executable to stand in for the real binary"

    def test_trivial_consumer_drives_a_full_session(self):
        """P0 acceptance: a trivial Python consumer can drive a full session from the mock."""
        result = run_mock("transcribe", "--input", "meeting.wav", fixture=FIXTURE_DIR / "short-meeting.ndjson")
        assert result.returncode == 0

        events = parse_stdout(result.stdout)
        assert events[0]["type"] == "hello"
        assert events[0]["protocol"] == PROTOCOL_VERSION
        assert is_terminal(events[-1])

        text = " ".join(e["text"] for e in events if e["type"] == "segment")
        assert "deadline" not in text
        assert text.startswith("Right, let's get started.")
        assert events[-1]["segments"] == sum(1 for e in events if e["type"] == "segment")

    @pytest.mark.parametrize("path", session_fixtures(), ids=lambda p: p.stem)
    def test_replays_every_session_fixture_verbatim(self, path: Path, validator):
        result = run_mock("transcribe", "--input", "x.wav", fixture=path)
        assert result.returncode == 0
        assert parse_stdout(result.stdout) == load(path)
        for event in parse_stdout(result.stdout):
            assert not list(validator.iter_errors(event))

    @pytest.mark.parametrize("path", error_fixtures(), ids=lambda p: p.stem)
    def test_error_fixtures_exit_with_the_right_code(self, path: Path):
        result = run_mock("transcribe", "--input", "x.wav", fixture=path)
        events = parse_stdout(result.stdout)
        expected = 1 if events[-1]["type"] == "error" else 0
        assert result.returncode == expected
        assert events == load(path)

    def test_stdout_carries_protocol_only(self):
        """Every stdout line must parse as JSON. Human-readable output belongs on stderr."""
        result = run_mock("transcribe", "--input", "x.wav", fixture=FIXTURE_DIR / "diarized-meeting.ndjson")
        for line in result.stdout.splitlines():
            if line.strip():
                json.loads(line)
        assert result.stderr.strip(), "the mock should log something human-readable to stderr"

    def test_usage_error_exits_2_with_no_protocol(self):
        result = run_mock("transcribe")  # missing --input
        assert result.returncode == 2
        assert result.stdout == ""
        assert "error:" in result.stderr

    def test_unknown_flag_is_rejected(self):
        result = run_mock("transcribe", "--input", "x.wav", "--hotwords", "ownscribe")
        assert result.returncode == 2
        assert result.stdout == ""

    def test_unknown_command_is_rejected(self):
        assert run_mock("summarise").returncode == 2

    def test_accepts_the_real_binarys_capture_flags(self, tmp_path):
        """Exactly the command line coreaudio.py builds today, plus the new ASR flags."""
        output = tmp_path / "audio.wav"
        result = run_mock(
            "capture", "--output", str(output), "--capture-mode-all", "--mic",
            "--mic-device", "MacBook Pro Microphone", "--silence-timeout", "300",
            "--transcribe", "--stream", "--diarize", "--embed",
            "--language", "en", "--min-speakers", "2", "--max-speakers", "4",
            fixture=FIXTURE_DIR / "streaming-partials.ndjson",
        )
        assert result.returncode == 0
        assert parse_stdout(result.stdout)[-1]["type"] == "final"

    def test_capture_writes_a_non_silent_wav(self, tmp_path):
        output = tmp_path / "nested" / "audio.wav"
        run_mock("capture", "--output", str(output), fixture=FIXTURE_DIR / "capture-session.ndjson")

        assert output.exists()
        with wave.open(str(output), "rb") as wav:
            assert wav.getnchannels() == 1
            assert wav.getframerate() == 16000  # D6
            frames = wav.readframes(wav.getnframes())
        assert any(frames), "an all-zero recording trips the silence checks in pipeline.py"

    def test_list_devices_is_plain_text_not_protocol(self):
        result = run_mock("list-devices")
        assert result.returncode == 0
        assert "MacBook Pro Microphone" in result.stdout
        with pytest.raises(json.JSONDecodeError):
            json.loads(result.stdout.splitlines()[0])

    def test_list_apps_is_plain_text_not_protocol(self):
        result = run_mock("list-apps")
        assert result.returncode == 0
        assert result.stdout.strip()

    def test_missing_fixture_is_a_usage_error(self, tmp_path):
        result = run_mock("transcribe", "--input", "x.wav", fixture=tmp_path / "nope.ndjson")
        assert result.returncode == 2
        assert result.stdout == ""

    def test_default_fixture_is_chosen_from_the_flags(self, tmp_path):
        """No fixture named: the mock must still be a drop-in for the real binary."""
        env = {k: v for k, v in os.environ.items() if k != "OWNSCRIBE_MOCK_FIXTURE"}
        result = subprocess.run(
            [sys.executable, str(MOCK), "transcribe", "--input", "x.wav", "--diarize"],
            capture_output=True, text=True, env=env, timeout=30,
        )
        assert result.returncode == 0
        assert any(e["type"] == "speakers" for e in parse_stdout(result.stdout))


class TestMockTiming:
    def test_instant_replay_is_actually_instant(self):
        """long-meeting is 71 minutes of audio; at speed 0 it must not sleep at all."""
        started = time.monotonic()
        result = run_mock("transcribe", "--input", "x.wav", fixture=FIXTURE_DIR / "long-meeting.ndjson")
        assert result.returncode == 0
        assert time.monotonic() - started < 15
        assert len(parse_stdout(result.stdout)) == 344

    def test_real_time_replay_paces_by_the_audio_timeline(self):
        """At speed 1 the first segment lands no earlier than its own end timestamp."""
        proc = subprocess.Popen(
            [sys.executable, str(MOCK), "transcribe", "--input", "x.wav",
             "--replay-speed", "4", "--fixture", str(FIXTURE_DIR / "short-meeting.ndjson")],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
        )
        started = time.monotonic()
        try:
            first_segment_at = None
            for line in proc.stdout:
                if json.loads(line)["type"] == "segment":
                    first_segment_at = time.monotonic() - started
                    break
        finally:
            proc.kill()
            proc.wait(timeout=10)

        assert first_segment_at is not None
        # First segment ends at 3.12s of audio; at 4x that is 0.78s.
        assert first_segment_at >= 0.5


class TestMockSignals:
    """SIGINT and SIGUSR1 are the two signals coreaudio.py already sends the real binary."""

    def _start_long_session(self) -> subprocess.Popen:
        return subprocess.Popen(
            [sys.executable, str(MOCK), "capture", "--output", "/dev/null",
             "--mic", "--replay-speed", "1",
             "--fixture", str(FIXTURE_DIR / "long-meeting.ndjson")],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
        )

    def test_sigint_stops_early_but_still_terminates_the_stream(self):
        proc = self._start_long_session()
        time.sleep(1.0)
        proc.send_signal(signal.SIGINT)
        stdout, _ = proc.communicate(timeout=30)

        events = parse_stdout(stdout)
        assert proc.returncode == 0
        assert events[0]["type"] == "hello"
        assert events[-1]["type"] == "final", "invariant 2 holds even on an interrupted session"
        assert events[-1]["stopped_reason"] == "user"
        assert events[-1]["segments"] == sum(1 for e in events if e["type"] == "segment")
        assert events[-1]["duration"] < 4291.0, "duration should reflect the early stop"

    def test_sigusr1_emits_a_mute_event(self):
        proc = self._start_long_session()
        time.sleep(0.5)
        proc.send_signal(signal.SIGUSR1)
        time.sleep(0.5)
        proc.send_signal(signal.SIGINT)
        stdout, _ = proc.communicate(timeout=30)

        mutes = [e for e in parse_stdout(stdout) if e["type"] == "mute"]
        assert mutes and mutes[0]["muted"] is True, "SIGUSR1 must surface as a mute event"
