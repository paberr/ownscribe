# ownscribe `next` — architecture and delivery plan

Branched from `main` at v0.14.0 (`0740776`).

This document is the single source of truth for the `next` effort.
**§5 work parcels** are self-contained briefs, and
**§4** says which may run at the same time.

Rules for anyone (human or agent) working from this document:

- Decisions in §2 are settled. Do not relitigate them in code review.
- Items marked **OPEN** are not settled — surface findings, do not invent an answer.
- Do not start a parcel whose dependencies have not landed. Parallelism is bounded by
  the graph in §4, not by how many agents are available.

---

## 1. Why

Transcription currently runs on WhisperX → faster-whisper → CTranslate2, and
**CTranslate2 has no Metal backend**. `whisperx_transcriber.py:47` hardcodes
`device = "cpu"` because that is the only valid option for that stack. On Apple
Silicon — the only platform where system audio capture works at all — the GPU and
Neural Engine are idle.

On a published M4 benchmark of eight Whisper implementations, this exact configuration
placed last by a factor of five. CoreML and MLX alternatives ran 5×–36× faster.

Moving ASR and diarization into the Swift helper that already exists does three things
at once:

1. Captures that speedup.
2. Deletes torch, torchaudio, whisperx, pyannote-audio and ctranslate2 from the Python
   dependency tree (~2 GB installed, plus 3–5 GB of model downloads).
3. Enables streaming transcription — which is the thing that makes a native UI worth
   building, and which is impossible in the current post-hoc architecture.

---

## 2. Decisions

### D1 — Apple Silicon only, macOS 14.2+

Intel is unsupported. Today `coreaudio.py:39` accepts `x86_64` and builds a release URL
for it, but the release workflow only ever uploads `ownscribe-audio-arm64`; the fetch
404s, is swallowed, and capture silently degrades to microphone-only. On `next` this
becomes an explicit, actionable error.

Floor is **macOS 14.2**: FluidAudio requires macOS 14, Core Audio taps require 14.2.

*Basis:* across 40 issues and PRs, not one mentions Linux or Windows. "Linux/Windows
audio capture backends" has been the first listed open contribution area in
CONTRIBUTING.md for six months at 95 stars and 44 forks, with no takers. The arm64
binary has 313 downloads.

### D2 — ASR is Parakeet TDT v3 via FluidAudio. WhisperX is removed entirely.

Not kept as an optional extra. This is a deliberate simplification with two accepted
regressions that **must** be handled explicitly, not silently:

- **Language coverage drops from 99 to 25** (European; English and German included).
  An unsupported language must produce a clear error naming the supported set — never
  a silent mistranscription.
- **`initial_prompt` and `hotwords` are removed.** FluidAudio's ASR API exposes token
  timings and confidence but no biasing, hotwords or prompt. These config keys must be
  detected and reported as removed, not ignored.

### D3 — Diarization is FluidAudio's pyannote Community-1

Same model family ownscribe already uses, so output quality is a known quantity.
Sortformer (≤4 speakers) and LS-EEND (streaming, ≤10 speakers) are noted as future
options, not part of this effort.

Speaker embedding extraction comes with it, which closes the long-standing "speaker
name assignment" item — scoped here as persistent speaker profiles so `SPEAKER_01`
can become a real name across meetings.

### D4 — The Python/Swift boundary moves

| Owns | Component |
|---|---|
| **Swift** (`ownscribe-core`) | Capture, VAD, ASR, diarization, speaker embeddings |
| **Python** | CLI, config, orchestration, summarization, output formatting, search |

They communicate over a **versioned NDJSON protocol** (§3). This is the load-bearing
decision: it is what lets most parcels proceed in parallel.

### D5 — Streaming is in scope

The protocol carries partial results from the first version, even where the initial
implementation is batch. Retrofitting streaming into a batch-shaped protocol later
would mean redoing every consumer.

### D6 — Capture at 16 kHz mono

Currently 24 kHz float32 (`AudioCapture.swift:18`), then ffmpeg resamples to 16 kHz for
Whisper. Capturing at 16 kHz directly removes the resample and cuts file size ~3×
(today: ~345 MB/hour, and `keep_recording` defaults to true).

### D7 — Summarization is unchanged

Stays Python + llama-cpp-python. Out of scope for `next`. The context-window and
chunking fixes are landing separately on `main` and will merge forward.

### D8 — `ask` / search is deprioritised

No changes on this branch. The O(n) LLM scan is a known issue and remains open.

### D9 — Distribution stays PyPI / uvx

`uvx ownscribe` with the Swift binary auto-downloading on first run, exactly as today,
arm64-only. See §7 for how this is expected to evolve — the protocol and bundle layout
are designed so that evolution is not a rewrite.

### D10 — Python dependency target

After `next`, the Python tree must contain **no** torch, torchaudio, whisperx,
pyannote-audio or ctranslate2. This is a hard acceptance criterion, verifiable with
`uv tree`.

---

## 3. The protocol — the linchpin

The Swift binary emits **newline-delimited JSON on stdout**, one object per line.

Two rules make this work, and both fix real problems in the current codebase:

1. **stdout carries protocol only.** Never a stray `print`.
2. **stderr is human-readable logs and is never parsed for control flow.** Today
   `coreaudio.py:157` scrapes stderr for `[SILENCE_WARNING]` and `[SILENCE_TIMEOUT]`
   sentinels, and `whisperx_transcriber.py` wraps everything in nested
   `redirect_stdout` calls with a `DownloadProgressWriter` that reverse-engineers
   progress bars printed by third-party libraries. **All of that machinery is deleted
   by this decision.**

### Event sketch

```jsonc
{"type":"hello","protocol":1,"capabilities":["asr","diarize","stream","embed"],
 "models":{"asr":"parakeet-tdt-0.6b-v3"}}

{"type":"progress","stage":"download","item":"parakeet-tdt-0.6b-v3",
 "fraction":0.42,"detail":"120 MB / 285 MB"}

{"type":"state","stage":"capturing|transcribing|diarizing","status":"begin|end"}

// streaming only; may be revised by a later segment covering the same span
{"type":"partial","start":12.4,"end":15.1,"text":"so the deadline is"}

{"type":"segment","id":37,"start":12.4,"end":16.0,
 "text":"So the deadline is Friday.","speaker":"SPEAKER_01",
 "words":[{"t":"So","s":12.4,"e":12.6,"c":0.98}]}

{"type":"speakers","count":3,"labels":["SPEAKER_00","SPEAKER_01"]}

{"type":"final","language":"en","duration":3612.5,"segments":842}

{"type":"error","code":"unsupported_language","message":"...","recoverable":false}
```

**Invariants:** `hello` is always first; exactly one `final` **or** `error` is last;
consumers must ignore unknown `type` values and unknown fields (forward compatibility);
`protocol` is an integer that Python feature-detects against.

### Why this unlocks parallelism

P0 also ships a **mock binary** that replays canned protocol output from fixtures at
realistic timing. With it, the Python backend (P2), the dependency purge (P5) and the
menu bar app (P6) can all be built and fully tested **before any Swift code exists**.

---

## 4. Dependency graph and parallelism

```
P7  eval harness ─────────────────────────────  no dependencies, start immediately
P8  docs / packaging ──────────────────────────  continuous, finalise last

P0  protocol + mock binary   [BLOCKING — nothing below starts until this lands]
     │
     ├─ P1  Swift: ASR ──────────── P4  Swift: streaming + 16 kHz capture
     ├─ P2  Python: core backend
     ├─ P3  Swift: diarization + speaker embeddings
     ├─ P5  Python: dependency purge, config & CLI surface
     └─ P6  Menu bar app
```

**Critical path:** `P0 → P1 → P4`. Adding agents does not shorten it. Everything else
fans out from P0 and can run genuinely concurrently.

**Integration checkpoints** — these are merge points, not parcels:

| Checkpoint | When | What it proves |
|---|---|---|
| **I1** | P1 + P2 land | Swap the mock for the real binary; transcription works end to end. |
| **I2** | I1 + P3 + P5 land | Diarization works; `uv tree` shows no torch. |
| **I3** | I2 + P4 land | Streaming partials arrive during recording. |
| **I4** | I3 + P6 land | Menu bar app drives a full session. |

**Recommended agent allocation after P0:** one each on P1, P2, P3, P5, P7. Hold P6
until the protocol has survived first contact in I1 — building a UI against a mock that
later changes shape is wasted work. P4 starts when P1 lands.

---

## 5. Work parcels

Each parcel below is written to be handed to an agent as-is. All of them inherit the
repo conventions: Python 3.12+, ruff line-length 120, `from __future__ import
annotations`, lazy imports for heavy dependencies, helpers return data while
orchestrators own `click.echo`. `PipelineProgress` in `progress.py` must not be
replaced or simplified. Tests accompany every parcel.

---

### P0 — Protocol specification and mock binary  `BLOCKING`

**Depends on:** nothing. **Blocks:** P1, P2, P3, P5, P6.

Everything else waits on this, so keep it small and land it fast. Half a day, one agent.

**Deliverables**

1. `docs/protocol.md` — the full event schema, invariants, versioning policy, and error
   code list. Expand the sketch in §3; that sketch is a starting point, not a spec.
2. `tests/fixtures/protocol/*.ndjson` — canned sessions: short meeting, long meeting,
   diarized multi-speaker, streaming with revised partials, model-download progress,
   and each error code.
3. `tools/mock-ownscribe-core` — an executable that replays a fixture with realistic
   timing, accepting the same CLI flags the real binary will. Must support replaying
   instantly (for tests) and in real time (for UI development).
4. A JSON Schema for each event type, used by both Swift and Python test suites so the
   two implementations cannot drift.

**Acceptance:** the mock is invocable exactly as the real binary will be, and a trivial
Python consumer can drive a full session from it.

**Notes:** design the error codes deliberately — `unsupported_language`,
`model_download_failed`, `no_audio_captured`, `permission_denied` and
`unsupported_architecture` are all known-needed. `permission_denied` matters: missing
Screen Recording permission currently surfaces as a silent all-zero recording caught
after the fact by `_check_audio_silence`.

---

### P1 — Swift: ASR via FluidAudio

**Depends on:** P0. **Blocks:** P4. **Parallel with:** P2, P3, P5, P6, P7.

**Goal:** `ownscribe-core transcribe --input FILE` emits protocol events on stdout and
exits 0. Standalone and testable with no Python involved.

**Deliverables**

1. FluidAudio (Apache 2.0) added to `swift/Package.swift`. Note it requires Swift 6.0+;
   confirm the current toolchain and `swift/build.sh` still work.
2. Parakeet TDT v3 wired up, with model download and caching, emitting `progress` events
   during the download.
3. Token timings mapped to the `words` array in `segment` events. FluidAudio's
   `ASRResult` exposes token timings; grouping tokens into words is on you, and P3 needs
   this to be right.
4. Language auto-detection, with a hard `unsupported_language` error outside the 25
   supported languages.

**Acceptance:** transcribing a fixture WAV produces a schema-valid session; a
non-European-language input errors cleanly rather than emitting garbage.

**OPEN — report findings, do not guess:**
- Exact model download mechanism, cache location, and total on-disk size.
- Whether word grouping from token timings is reliable enough for speaker assignment.
- Whether any HuggingFace token or gated-terms acceptance is required. If FluidAudio's
  CoreML weights are ungated, the entire `hf_token` flow and its README section
  disappear — a significant onboarding win worth confirming early.

---

### P2 — Python: `ownscribe-core` transcriber backend

**Depends on:** P0 (mock). **Parallel with:** P1, P3, P5, P6, P7.

Build entirely against the mock. Do not wait for P1.

**Goal:** a `Transcriber` implementation that drives the binary and returns
`TranscriptResult`.

**Deliverables**

1. `src/ownscribe/transcription/core_transcriber.py` implementing the existing
   `Transcriber` ABC — subprocess lifecycle, NDJSON parsing, protocol version check,
   mapping to `Segment`/`Word`/`TranscriptResult`.
2. Wired into `_create_transcriber` in `pipeline.py`.
3. `progress` and `state` events driven into the existing `PipelineProgress` TUI. This
   replaces the stdout-scraping path — the TUI's behaviour must not regress.
4. Robust failure handling: binary missing, protocol version mismatch, malformed line,
   process dies mid-session, non-zero exit.

**Acceptance:** full test coverage against every P0 fixture, including all error paths.
**This module must not import torch, whisperx or pyannote** — assert it in a test.

---

### P3 — Swift: diarization and speaker embeddings

**Depends on:** P0. **Parallel with:** P1, P2, P5, P6, P7.

Independent of P1 — different FluidAudio subsystem. Can be developed against fixture
audio and merged with P1's output at I2.

**Deliverables**

1. pyannote Community-1 diarization behind a `--diarize` flag, emitting `speaker` fields
   on segments plus a `speakers` summary event.
2. `min_speakers` / `max_speakers` honoured (they exist in `DiarizationConfig` today).
3. Speaker embedding extraction exposed, so profiles can be persisted later.
4. Diarization progress driven through `progress` events. Today this is a pyannote hook
   wired into `progress.diarization_hook`; the TUI's sub-step display must keep working.

**Acceptance:** on a multi-speaker fixture, diarization quality is **compared against the
current pyannote output** and the comparison is written down. Same model family, so a
large divergence means the CoreML export or the integration is wrong.

**OPEN:** whether `hf_token` is still required (see P1). If not, remove the config key,
the env var override and the README section.

---

### P4 — Swift: streaming and 16 kHz capture

**Depends on:** P1. **Parallel with:** P3, P5, P6, P7. **On the critical path.**

The hardest parcel. Do not start it before P1 lands.

**Deliverables**

1. Capture switched to 16 kHz mono (D6) — `kSystemAudioSampleRate` and the merge path in
   `mergeAudioFiles`. Verify Core Audio taps deliver 16 kHz cleanly rather than forcing a
   resample inside the binary; if not, resample once at capture.
2. Streaming ASR via FluidAudio's `SlidingWindowAsrManager` or `StreamingEouAsrManager`,
   emitting `partial` events during capture.
3. A defined revision model: how a `partial` is superseded by a later `segment`, and what
   a consumer must do to render it correctly. Specify this in `docs/protocol.md`.
4. Capture and ASR running concurrently without dropping audio. Recording remains the
   hard real-time constraint — **ASR must never be able to stall capture.**

**Acceptance:** a full-length session produces partials during recording and a final
transcript that matches the batch path on the same audio.

**Preserve:** the existing capture hardening is hard-won and must not regress — mid-
recording route changes (earbuds), voice-processing format flips (Meet), display sleep
killing the stream, and host-time alignment of the system and mic tracks. Read
`AudioCapture.swift` before touching it. Its tests are the `@pytest.mark.hardware` set,
which CI skips, so it is easy to break silently.

---

### P5 — Python: dependency purge, config and CLI surface

**Depends on:** P0. **Parallel with:** P1, P2, P3, P4, P6, P7.

The parcel that realises D10 and makes the regressions in D2 visible to users.

**Deliverables**

1. Remove `whisperx`, `pyannote-audio`, `torchaudio` from `pyproject.toml`; delete
   `transcription/whisperx_transcriber.py` and its tests.
2. Delete `audio/sounddevice_recorder.py` and the `sounddevice` dependency. With ASR in
   the binary, a fallback that captures audio nothing can transcribe is not a fallback.
   **Mic-only capture must move into the binary first** (coordinate with P1/P4) —
   `--device "MacBook Pro Microphone"` is a documented workflow and must keep working.
3. Config migration: `[transcription] model` values change (Parakeet variants, not
   `tiny`/`base`/`small`/`large-v3`); `initial_prompt` and `hotwords` are removed;
   `[diarization] hf_token` likely removed. `_merge_toml` in `config.py` currently drops
   unknown keys silently — make it **warn on unknown and explain removed keys**, so users
   who configured `hotwords` find out instead of quietly losing it.
4. Architecture gate: refuse to run on non-arm64 with a clear message (D1).
5. `ownscribe apps` currently prints "binary not found" off macOS; align all subcommands
   with the arm64-only reality.

**Acceptance:** `uv tree` contains no torch, torchaudio, whisperx, pyannote-audio or
ctranslate2. Fresh-install size recorded in the PR description — this is one of the
headline numbers for the release.

---

### P6 — Menu bar app

**Depends on:** P0. Hold until **I1** — see §4.

**Goal:** a SwiftUI `MenuBarExtra` app, a second product in the same SwiftPM package,
consuming the same protocol.

**Deliverables**

1. Recording indicator with elapsed time; start/stop.
2. Live transcript view fed by `partial` events (degrades gracefully to "transcribing…"
   when streaming is unavailable, so this does not block on P4).
3. Mic mute toggle. **This already exists end to end** — `toggle_mute()` signals the
   binary with `SIGUSR1`, which zeroes the mic buffers in the tap — but is reachable only
   by pressing `m` in an interactive TTY (`pipeline.py`), so it is unavailable whenever
   ownscribe runs non-interactively. Expose it properly.
4. Recent meetings list linking to output folders.

**OPEN:** how the app invokes summarization. It can drive `ownscribe-core` directly for
capture and ASR, but summarization lives in Python. Either shell out to the `ownscribe`
CLI or ship the app as a front-end that requires the CLI installed. Decide before
building, and record the decision here — it determines the packaging story in §7.

---

### P7 — Evaluation harness

**Depends on:** nothing. **Start immediately, in parallel with P0.**

This parcel de-risks D2, which is now a one-way door: WhisperX is being deleted, so if
Parakeet is worse on real meeting audio we need to know **before** the code is gone, not
after.

Parakeet's headline benchmarks are on clean read speech. Meeting audio is overlapping
speakers over compressed VoIP codecs with variable mic quality — a materially different
distribution.

**Deliverables**

1. A script that runs a corpus of recordings through both the current WhisperX path and
   the new backend, reporting per-file and aggregate wall-clock and WER/CER.
2. Diarization comparison (DER where reference labels exist; speaker-count agreement
   otherwise).
3. A results document committed to the branch, including at least one long (>1 hour)
   multi-speaker recording and one non-English recording.

**Acceptance:** results are written down and reviewed before P5 deletes WhisperX.

**Note:** real meeting recordings are sensitive. Keep audio and transcripts out of the
repo; commit only aggregate metrics, and put the corpus path behind an env var.

---

### P8 — Documentation and packaging

**Depends on:** continuous; finalise after P1, P2, P5. **Parallel with:** everything.

**Deliverables**

1. README rewrite: Apple Silicon + macOS 14.2 requirements, new model names, removal of
   the HF token section if P1/P3 confirm it is unnecessary.
2. **Migration guide** — the user-facing face of D2. Must state plainly: which languages
   are no longer supported, that `initial_prompt` and `hotwords` are gone, and what
   config keys changed. Someone upgrading and finding their language unsupported should
   hit documentation, not confusion.
3. `pyproject.toml` classifiers and CONTRIBUTING.md updated: remove the
   "Linux/Windows audio capture backends" contribution area (D1) and the
   "Additional LLM backends" item if it no longer reflects intent.
4. Release workflow: arm64 only, and confirm whether ASR/diarization models are bundled
   or downloaded on first run.

---

## 6. Risks

| Risk | Severity | Mitigation |
|---|---|---|
| Parakeet underperforms on real meeting audio | **High** — D2 is a one-way door | P7, before P5 deletes WhisperX |
| Streaming ASR stalls capture under load | **High** — corrupts the recording | Hard isolation in P4; capture stays real-time priority |
| CoreML diarization diverges from current pyannote | Medium | Explicit comparison in P3's acceptance |
| Capture hardening regresses during the 16 kHz change | Medium | CI skips hardware tests; manual test matrix in P4 |
| Protocol churns after consumers are built | Medium | P0 lands first; P6 held until I1 |
| FluidAudio model licensing or gated downloads | Low–Medium | Resolve in P1 as an OPEN item |
| Losing 74 languages costs real users | Accepted (D2) | Clear error + migration guide (P8) |

---

## 7. How distribution is expected to evolve

D9 keeps PyPI/uvx for now. The likely path, and what to preserve so it stays cheap:

1. **Today / `next`:** `uvx ownscribe`, binary auto-downloaded from GitHub Releases.
   Screen Recording permission attaches to the user's terminal.
2. **Next:** a signed, notarized `.app` bundling `ownscribe-core` and the menu bar UI,
   distributed via Homebrew cask. Permission attaches to ownscribe itself, which is both
   better UX and a **migration step for every existing user** — and the failure mode of
   a missing grant is a silent all-zero recording, so it needs a first-run check that
   fails loudly.
3. **Possibly:** the CLI as a thin client of the bundled binary, so both channels ship
   the same core.

To keep step 2 cheap, two things must hold from the start: the binary must be
relocatable (no assumptions about living next to a Python venv — see `_BINARY_CANDIDATES`
in `coreaudio.py`, which currently probes `sys.prefix`), and P6 must resolve its OPEN
item about how the UI reaches summarization.

---

## 8. Migration summary for users

| Change | Impact |
|---|---|
| Apple Silicon only | Intel Macs unsupported; clear error instead of silent mic-only capture |
| Whisper → Parakeet TDT v3 | 25 European languages instead of 99 |
| `initial_prompt`, `hotwords` removed | No equivalent in the new backend |
| `[transcription] model` values changed | `tiny`/`base`/`small`/`large-v3` no longer valid |
| `[diarization] hf_token` | Likely unnecessary — confirm in P1/P3 |
| Recordings now 16 kHz | ~3× smaller; existing files still transcribe |
| Install size | Substantially smaller — no PyTorch |
