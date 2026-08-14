# Protocol v1 JSON Schemas

Machine-readable form of [`docs/protocol.md`](../../../docs/protocol.md). Both the Python
test suite and the Swift test suite validate against these files, so the two implementations
cannot drift apart silently.

`event.json` is the entry point: it dispatches on `type` and validates the matching event
schema. Validating a whole session means validating every line against `event.json`, then
checking the ordering invariants (which JSON Schema cannot express) separately.

## These are test-time schemas, not runtime validators

Every event schema sets `additionalProperties: false`. That is deliberate, and it is the
opposite of what a *consumer* must do at runtime.

- **At test time**, strictness is the point: it catches a typo'd field name in a fixture or
  an implementation before it reaches anyone.
- **At runtime**, protocol invariant 4 requires consumers to ignore unknown fields and
  unknown event types, so that additive changes stay free. A consumer that ran these schemas
  against live output would reject a newer binary that is behaving perfectly correctly.

So: validate fixtures and your own emitted output against these. Never validate a peer's
input against them.

When an additive change lands, the schema is updated in the same commit. `protocol` stays
`1` — see `docs/protocol.md` §8.
