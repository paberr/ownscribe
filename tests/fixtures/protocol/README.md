# Protocol fixtures

Canned `ownscribe-core` sessions, one NDJSON file per session. They are the input to
`tools/mock-ownscribe-core`, and they are what `tests/test_protocol.py` validates against
[`schema/protocol/v1/`](../../../schema/protocol/v1/).

Their real job is to let P2 (Python backend), P5 (dependency purge) and P6 (menu bar app)
be built and fully tested before any Swift exists.

## Sessions

| Fixture | Covers |
|---|---|
| `short-meeting.ndjson` | The ordinary path: four segments with word timings, no diarization. |
| `long-meeting.ndjson` | Volume: 300 segments, 344 events, 71 minutes of audio timeline, diarized. |
| `diarized-meeting.ndjson` | Three speakers, all four diarization sub-stages, speaker embeddings. |
| `streaming-partials.ndjson` | Partials revised by later partials and superseded by segments; mute toggles mid-session. |
| `model-download.ndjson` | First-run model download progress, two models, then a CoreML compile step with no numeric progress. |
| `capture-session.ndjson` | Capture with no ASR — the `hello` advertises only `capture`. |

## Errors

`errors/` holds one fixture per error code in `defs.json`. `test_every_error_code_has_a_fixture`
fails if a code is added without one.

The two **recoverable** codes — `no_audio_captured` and `device_changed` — are the ones worth
reading first: they appear mid-stream and the session still ends in a normal `final`. Every
other code is terminal and the stream stops there, with a stage left open (which is legal;
the consumer calls `progress.fail()`).

## Regenerating

`short-meeting`, `diarized-meeting`, `streaming-partials`, `model-download` and
`capture-session` are handwritten — edit them directly.

`long-meeting.ndjson` and everything under `errors/` were generated deterministically. They
are committed as data, not built at test time, so that a fixture change shows up in review as
a diff. Edit them directly too; the invariant tests will catch anything inconsistent.

## Adding a fixture

Drop it in and the parametrised tests pick it up. It must satisfy the invariants in
`docs/protocol.md` §2 — `hello` first, exactly one terminal event last, segment ids strictly
increasing, `final.segments` matching the segments actually emitted.
