# Optional machine-readable history

`PLAN.md` and `agent-notes/STATE.md` are the current handoff state. `state.json` is an optional structured snapshot; it may describe an older checkpoint. Update it only when a task needs a machine-readable state record.

`events.jsonl` is optional and append only. If writing an event, each non-empty line must validate against `event.schema.json`. Use paths to durable evidence instead of copying large logs into an event.

When maintaining and correcting this history, append a new event and set `supersedes` to the earlier `event_id`; never mutate old lines.
