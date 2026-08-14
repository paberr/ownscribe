# The `ownscribe-core` protocol

Version **1**.

`ownscribe-core` is the Swift helper that owns capture, VAD, ASR, diarization and speaker
embeddings (NEXT.md D4). It reports everything it does as newline-delimited JSON on stdout.
This document is the contract between it and every consumer: the Python CLI (P2), the menu
bar app (P6), and the test suites on both sides.

Expanded from the sketch in NEXT.md §3. Where this document and that sketch disagree, this
document wins — §3 says explicitly that the sketch is a starting point, not a spec. Every
deviation is called out in [§9](#9-deviations-from-the-nextmd-sketch) with its reason.

---

## 1. Transport

- One JSON object per line on **stdout**, UTF-8, `\n`-terminated. No pretty-printing, no
  blank lines, no trailing commas, no comments.
- **stdout carries protocol only.** A stray `print` is a protocol violation, not a cosmetic
  bug. Anything a human should read goes to stderr.
- **stderr is human-readable logging and is never parsed for control flow.** Consumers may
  display it, capture it, or discard it. They must never branch on its contents.
- The binary flushes after every event. A consumer reading line-by-line sees events as they
  happen; this is what makes `partial` useful.
- Line length is unbounded in principle. Consumers must not assume a maximum.

The second rule deletes three pieces of existing machinery:

| Deleted | Replaced by |
|---|---|
| `[SILENCE_WARNING]` scraped from stderr (`coreaudio.py:159`) | `error` with code `no_audio_captured`, `recoverable: true` |
| `[SILENCE_TIMEOUT]` scraped from stderr (`coreaudio.py:161`) | `final.stopped_reason == "silence_timeout"` |
| `[MIC_MUTED]` / `[MIC_UNMUTED]` filtered as noise (`coreaudio.py:165`) | `mute` event |

…and on the Python side, the `redirect_stdout` nesting and `DownloadProgressWriter`
reverse-engineering of third-party progress bars in `whisperx_transcriber.py` (see
[§6](#6-progress-and-the-tui)).

### Exit codes

| Exit | Meaning |
|---|---|
| 0 | The stream ended with `final`. |
| 1 | The stream ended with a terminal `error`. The `error` event carries the detail. |
| 2 | Usage error — bad flags, before any `hello`. Nothing protocol-shaped was emitted. |
| other / signal | Abnormal termination. Consumers treat this as `process_died` regardless of what stdout contained. |

Exit code 2 is the only case where a consumer may find no `hello` on stdout. It exists so
that argument parsing does not have to succeed before the protocol can start.

---

## 2. Session shape

```
hello                                        ← always first, exactly once
  ( state | progress | partial | segment | speakers | mute | error(recoverable) )*
final | error(terminal)                      ← always last, exactly one
```

### Invariants

These are the guarantees a consumer may rely on, and the rules an implementation must not
break. Both test suites assert them.

1. **`hello` is the first line of every session**, and appears exactly once.
2. **The stream ends with exactly one terminal event**: either `final`, or an `error` with
   `recoverable: false`. Never both, never neither.
3. A **`recoverable: true` `error` is not terminal.** It may appear anywhere after `hello`,
   any number of times, and the session continues. It reports a degradation the consumer
   should surface, not a stop.
4. **Consumers must ignore unknown `type` values and unknown fields.** This is what makes
   additive protocol changes free. A consumer that rejects an unrecognised event is broken.
5. **`protocol` is an integer**, feature-detected by the consumer (see [§8](#8-versioning)).
6. Timestamps (`start`, `end`) are **seconds from the start of the audio**, as floats.
   Monotonic within a stream. `start <= end`.
7. `segment.id` is a **non-negative integer, strictly increasing** within a session. It is
   not necessarily contiguous.
8. If the process exits without satisfying invariant 2, the consumer must treat the session
   as failed (`process_died`) even if a partial transcript was accumulated.

Invariant 8 matters: a truncated NDJSON stream is indistinguishable from a successful short
one without it.

---

## 3. Events

Every event has a `type`. Fields not marked optional are always present.

### `hello`

Always first. Lets the consumer feature-detect before committing to a session.

```jsonc
{"type":"hello","protocol":1,"binary_version":"0.15.0",
 "capabilities":["asr","diarize","embed","stream","capture"],
 "models":{"asr":"parakeet-tdt-0.6b-v3","diarization":"pyannote-community-1"}}
```

| Field | Type | Notes |
|---|---|---|
| `protocol` | int | This document describes `1`. |
| `binary_version` | string | The helper's own version. Informational; never branch on it — branch on `capabilities`. |
| `capabilities` | string[] | Sorted, unique. See below. |
| `models` | object | Model identifiers actually loaded, keyed by role. Absent keys mean that role is not active this session. |

Capabilities:

| Capability | Means |
|---|---|
| `capture` | Can record system and/or mic audio. |
| `asr` | Can transcribe. |
| `diarize` | Can assign speakers. |
| `embed` | Can emit speaker embeddings. |
| `stream` | Can emit `partial` events during capture. |

`capabilities` is how P2 degrades gracefully while P4 is still in flight: a binary from P1
advertises `["asr","capture"]` and the consumer simply never waits for a `partial`.

### `state`

Marks a stage beginning or ending. This is the event that drives the TUI checklist.

```jsonc
{"type":"state","stage":"transcribing","status":"begin"}
{"type":"state","stage":"diarizing","substage":"embeddings","status":"begin"}
{"type":"state","stage":"diarizing","substage":"embeddings","status":"end"}
```

| Field | Type | Notes |
|---|---|---|
| `stage` | string | See the stage vocabulary in [§6](#6-progress-and-the-tui). |
| `substage` | string | Optional. Only meaningful for `diarizing`. |
| `status` | `"begin"` \| `"end"` | |

Rules:

- Every `begin` is eventually matched by an `end` **or** by the terminal event. A consumer
  must close open stages when the session terminates; it must not hang waiting.
- **`capturing` is session-scoped and may overlap any processing stage.** With `--transcribe
  --stream`, capture and ASR run concurrently by design (P4), so `capturing` and
  `transcribing` are open at the same time.
- **The processing stages — `preparing_models`, `transcribing`, `diarizing` — do not overlap
  one another.** A `begin` for one implies `end` for any other still open. This mirrors
  `PipelineProgress.begin()`, which already auto-completes an active sibling at the same
  indent level.
- A `substage` lives inside its `stage` and is subject to the same rule among substages.

### `progress`

Fractional progress within the current stage. Purely informational — a consumer that drops
every `progress` event still gets a correct transcript.

```jsonc
{"type":"progress","stage":"preparing_models","item":"parakeet-tdt-0.6b-v3",
 "fraction":0.42,"bytes_done":125829120,"bytes_total":298844160}
```

| Field | Type | Notes |
|---|---|---|
| `stage` | string | Same vocabulary as `state.stage`. |
| `substage` | string | Optional. |
| `item` | string | Optional. What is being worked on — a model name, a filename. |
| `fraction` | float | Optional. `0.0`–`1.0`, clamped by the consumer. Absent means indeterminate: show a spinner, not a bar. |
| `bytes_done`, `bytes_total` | int | Optional, but both-or-neither. Present for downloads. |
| `detail` | string | Optional free-text override for display. Use sparingly — see below. |

**Send structured bytes, not a formatted string.** `progress.py` already owns the formatting
(`format_download_progress`, `_human_bytes`), and P6 renders in a menu bar where "120 MB /
285 MB" is the wrong shape. `detail` exists for the cases that genuinely have no numeric
form; it is not the normal path.

Rate: at most ~10 `progress` events per second per stage. The TUI redraws at 10 Hz
(`progress.py:_INTERVAL`), so a faster rate is pure overhead. Emit on meaningful change,
not on a timer.

### `partial`

A provisional transcript fragment, emitted during capture when `stream` is advertised. May
be revised or withdrawn. See [§4](#4-the-revision-model) for exactly what that means.

```jsonc
{"type":"partial","start":12.4,"end":15.1,"text":"so the deadline is"}
```

| Field | Type | Notes |
|---|---|---|
| `start`, `end` | float | Seconds. The span this fragment covers. |
| `text` | string | Provisional. Never treat as final output. |

A `partial` never carries `speaker` or `words`. Diarization runs behind the streaming ASR;
attributing a speaker to unstable text would produce visible flicker for no gain.

### `segment`

A final, non-revisable transcript segment. This is the output.

```jsonc
{"type":"segment","id":37,"start":12.4,"end":16.0,
 "text":"So the deadline is Friday.","speaker":"SPEAKER_01",
 "words":[{"text":"So","start":12.4,"end":12.6,"confidence":0.98}]}
```

| Field | Type | Notes |
|---|---|---|
| `id` | int | Non-negative, strictly increasing. |
| `start`, `end` | float | Seconds. |
| `text` | string | Trimmed. May be empty only if `words` is also empty (a silence-only segment); consumers may drop those. |
| `speaker` | string \| null | Optional. `null` or absent when diarization is off. Label format `SPEAKER_NN`, or a resolved profile name once speaker profiles land. |
| `words` | object[] | Optional; empty array when word timings are unavailable. |

Word objects:

| Field | Type | Notes |
|---|---|---|
| `text` | string | One word. |
| `start`, `end` | float | Seconds. |
| `confidence` | float | Optional. `0.0`–`1.0`. |
| `speaker` | string | Optional. Per-word speaker, when it differs from the segment's. |

These map onto `transcription/models.py` without a translation layer:
`Word(text, start, end, speaker, score=confidence)` and
`Segment(text, start, end, speaker, words)`.

### `speakers`

Emitted once, after diarization completes, when `--diarize` is active.

```jsonc
{"type":"speakers","count":3,"labels":["SPEAKER_00","SPEAKER_01","SPEAKER_02"],
 "embeddings":[{"label":"SPEAKER_00","vector":[0.013,-0.42],"dim":2}]}
```

| Field | Type | Notes |
|---|---|---|
| `count` | int | Number of distinct speakers found. |
| `labels` | string[] | Sorted. Every label appearing on a `segment` appears here. |
| `embeddings` | object[] | Optional. Present only when `--embed` is passed. |

Embedding objects carry `label`, `vector` (float[]) and `dim` (int, `== vector.length`).
`dim` is redundant on purpose: it lets a consumer validate a vector cheaply and lets P3
change embedding size without consumers guessing. This is the hook for persistent speaker
profiles (D3); nothing in protocol 1 consumes it.

### `mute`

Mic mute state changed, in response to `SIGUSR1`.

```jsonc
{"type":"mute","muted":true}
```

Today this is `[MIC_MUTED]` on stderr, filtered out as noise by `coreaudio.py`, and mute is
reachable only by pressing `m` in an interactive TTY (`pipeline.py`) — so it is unavailable
whenever ownscribe runs non-interactively. As an event it becomes observable, which is what
P6 deliverable 3 needs. The signal-based *input* is unchanged.

### `final`

Terminal. Exactly one per successful session.

```jsonc
{"type":"final","language":"en","duration":3612.5,"segments":842,
 "stopped_reason":"user","audio_path":"/Users/x/ownscribe/2026-08-14_1030_standup/audio.wav"}
```

| Field | Type | Notes |
|---|---|---|
| `language` | string | BCP-47-ish code, detected or forced. Empty when not applicable. |
| `duration` | float | Audio duration in seconds. |
| `segments` | int | Count of `segment` events emitted. Consumers should assert this matches what they received — it is a cheap truncation check. |
| `stopped_reason` | string | Optional; capture sessions only. `"user"`, `"silence_timeout"`, or `"eof"`. |
| `audio_path` | string | Optional; capture sessions only. Absolute path to the written recording. |

`stopped_reason: "silence_timeout"` replaces the `[SILENCE_TIMEOUT]` stderr sentinel. It is
**not** an error: the recording succeeded and auto-stop is a configured feature
(`audio.silence_timeout`, default 300s).

### `error`

```jsonc
{"type":"error","code":"unsupported_language","recoverable":false,
 "message":"Detected language 'ja' is not supported.",
 "detail":{"detected":"ja","supported":["en","de","fr"]}}
```

| Field | Type | Notes |
|---|---|---|
| `code` | string | From the closed set in [§5](#5-error-codes). |
| `recoverable` | bool | `false` = terminal, the stream ends here. `true` = a warning; the session continues. |
| `message` | string | Human-readable, complete sentence, safe to show a user verbatim. |
| `detail` | object | Optional, code-specific structured data. Consumers must tolerate its absence. |

`message` is for humans; `code` is for control flow. A consumer must never match on
`message` text.

---

## 4. The revision model

Streaming means the consumer sees text before it is correct. The rule is deliberately
simple, because every consumer has to implement it.

**A `partial` is provisional and owns no ground.** The consumer keeps at most one
*pending region*, defined by the lowest `start` among partials not yet superseded.

1. A `partial` **supersedes** any earlier `partial` whose span overlaps it. Overlap is
   `a.start < b.end && b.start < a.end`.
2. A `segment` **supersedes every `partial`** whose span overlaps it, and those partials are
   discarded — their text is replaced by the segment's, not appended to it.
3. A `segment` never supersedes another `segment`. Segments are final.
4. Partials arrive in non-decreasing `start` order. A consumer may drop any `partial` whose
   `start` is below the end of the last emitted `segment`; it is stale.

Rendering, concretely: display all `segment` text in `id` order, then append the text of any
live partials in `start` order, visually distinguished (dimmed, italic). On a new `segment`,
drop the overlapping partials before appending. A consumer that ignores `partial` entirely
and renders only `segment` events is correct — just not live.

> **P4 owns the final word here.** This section is written to be sufficient for P6 to build
> against and for P2 to test against; P4 must confirm it against `SlidingWindowAsrManager`
> or `StreamingEouAsrManager` behaviour and amend it if the real revision granularity
> differs. If it does, the amendment lands here before P4's implementation.

---

## 5. Error codes

Closed set. Adding a code is an additive change ([§8](#8-versioning)); consumers must treat
an unrecognised code as a generic failure and show `message`.

| Code | Recoverable | When |
|---|---|---|
| `unsupported_architecture` | no | Not Apple Silicon (D1). `detail`: `{"arch":"x86_64"}`. |
| `unsupported_os` | no | macOS below 14.2. `detail`: `{"version":"13.6","required":"14.2"}`. |
| `permission_denied` | no | Screen Recording or Microphone permission missing. `detail`: `{"permission":"screen_recording"}`. |
| `unsupported_language` | no | Detected or requested language outside the supported set (D2). `detail`: `{"detected":…,"supported":[…]}`. |
| `model_download_failed` | no | Download or cache write failed. `detail`: `{"model":…,"url":…}`. |
| `input_not_found` | no | `--input` path missing or unreadable. |
| `invalid_input` | no | Input exists but is not decodable audio. |
| `capture_failed` | no | Capture could not start, or died mid-session unrecoverably. |
| `no_audio_captured` | **yes** | Capture produced silence. The recording still exists. |
| `device_changed` | **yes** | Audio route changed mid-recording and was re-established. |
| `internal_error` | no | Anything unclassified. `message` must still be actionable. |

`permission_denied` is the one that matters most today: a missing Screen Recording grant
currently surfaces as a silent all-zero recording, caught after the fact by
`_check_audio_silence` in `pipeline.py`. Detecting it up front and failing loudly is a
user-visible improvement, and NEXT.md §7 flags it again as the failure mode of the future
`.app` migration.

`no_audio_captured` is recoverable because the user's audio file is still on disk and they
may still want it. It replaces the `[SILENCE_WARNING]` sentinel, and it is what lets the CLI
skip its own `_check_audio_silence` pass.

---

## 6. Progress and the TUI

`PipelineProgress` in `progress.py` must not be replaced or simplified (NEXT.md §5). So the
stage vocabulary **is** its step-key vocabulary — no translation table to drift out of sync:

| `stage` | `PipelineProgress` key | Emitted by |
|---|---|---|
| `capturing` | *(no step; drives the recording UI)* | capture sessions |
| `preparing_models` | `preparing_models` | before ASR/diarization when a download or load is needed |
| `transcribing` | `transcribing` | ASR |
| `diarizing` | `diarizing` | diarization |

…and `substage`, valid only under `diarizing`, uses the existing sub-step keys in the order
`PipelineProgress` declares them:

`segmentation` → `speaker_counting` → `embeddings` → `clustering`

That ordering is not decorative: `PipelineProgress.begin()` auto-completes the active
sibling at the same indent level, so emitting them out of order silently marks steps done
that never ran. P3 must emit them in this order.

This is why `state` carries `substage` at all — the NEXT.md §3 sketch has no way to express
a diarization sub-step, but P3 deliverable 4 requires the sub-step display to keep working.

The mapping a consumer implements is mechanical:

| Event | Call |
|---|---|
| `state{stage,status:"begin"}` | `progress.begin(key)` |
| `state{stage,status:"end"}` | `progress.complete(key)` |
| `progress{stage,fraction}` | `progress.update(key, fraction)` |
| `progress{stage,bytes_done,bytes_total}` | `progress.set_detail(key, format_download_progress(...))` |
| terminal `error` | `progress.fail(key)` for the open stage |

`capturing` has no checklist step because recording is displayed by the live recording UI,
not the checklist. Consumers that have no recording UI ignore it.

---

## 7. CLI surface

The binary keeps its existing verbs and adds ASR. Flags that exist today keep their exact
spelling — `coreaudio.py` builds these command lines and P2 inherits them.

```
ownscribe-core capture    --output FILE [--mic] [--mic-device NAME]
                          [--capture-mode-all] [--silence-timeout SECONDS]
                          [--transcribe] [--stream] [--diarize] [--embed]
                          [--language CODE] [--min-speakers N] [--max-speakers N]
ownscribe-core transcribe --input FILE [--diarize] [--embed] [--language CODE]
                          [--min-speakers N] [--max-speakers N]
ownscribe-core list-devices
ownscribe-core list-apps
```

| Flag | Notes |
|---|---|
| `--output` / `-o` | Capture destination. Unchanged. |
| `--mic`, `--mic-device` | Unchanged. `--mic-device` must not re-enable a mic disabled by config — that logic stays in Python (`coreaudio.py:106`). |
| `--capture-mode-all` | Unchanged. Absent means the source picker. |
| `--silence-timeout` | Unchanged; seconds, `0` disables. |
| `--input` | `transcribe` only. Any format the system decoder accepts. |
| `--transcribe` | `capture` only. Transcribe while recording rather than post-hoc. |
| `--stream` | Requires `--transcribe`. Emit `partial` events. Ignored with a `recoverable` warning if `stream` is not in `capabilities`. |
| `--diarize` | Assign speakers. Emits `speakers`. |
| `--embed` | Requires `--diarize`. Include embeddings in `speakers`. |
| `--language` | Force a language instead of detecting. Outside the supported set → `unsupported_language`. |
| `--min-speakers`, `--max-speakers` | `0` = unconstrained. Mirrors `DiarizationConfig`. |

`list-devices` and `list-apps` predate this protocol and still print plain text for humans.
They are not protocol commands and emit no `hello`. Changing that is out of scope for
protocol 1 — nothing needs it.

---

## 8. Versioning

`hello.protocol` is a single integer. There is no minor version.

**Additive changes do not bump it.** Adding an event type, an optional field, an error code,
or a capability is free, because invariant 4 requires consumers to ignore what they do not
recognise. New *behaviour* is gated behind a `capabilities` entry, not a version bump.

**Breaking changes bump it.** Removing or renaming a field, changing a field's type or
units, changing an invariant, or changing the meaning of an existing value.

Consumers declare a supported range and check `hello.protocol` against it:

- Below the minimum → refuse with a clear "binary too old, upgrade" message.
- Above the maximum → refuse with a clear "binary too new, upgrade ownscribe" message.

Both are consumer-side failures, not `error` events — the binary is behaving correctly; the
consumer cannot understand it. P2 must cover both directions, since the binary
auto-downloads from GitHub Releases (D9) and can therefore get ahead of an installed CLI.

---

## 9. Deviations from the NEXT.md sketch

Recorded so review does not have to rediscover the reasoning. All are cheap to reverse.

1. **Word fields are spelled out** — `{"text","start","end","confidence"}` rather than
   `{"t","s","e","c"}`. The compact form saves roughly 200 KB across a one-hour meeting on a
   local pipe, and costs legibility in the fixtures committed to this repo and in every
   debugging session across three consumers. The names also match `models.py` exactly.
2. **`state` gains `substage`** — the sketch cannot express diarization sub-steps, which P3
   deliverable 4 requires ([§6](#6-progress-and-the-tui)).
3. **`progress` carries structured bytes rather than a formatted `detail` string** — the
   formatting already exists in `progress.py`, and P6 needs a different rendering.
4. **`mute` event added** — closes the gap P6 deliverable 3 describes and removes the third
   stderr sentinel.
5. **`stopped_reason` on `final`** — silence timeout is a successful outcome, not an error,
   so it does not belong in the error taxonomy.
6. **The `recoverable` invariant is stated precisely** — the sketch has both a `recoverable`
   flag and an "exactly one `final` or `error` is last" rule, which are in tension. Resolved
   in favour of both: recoverable errors are non-terminal warnings.
