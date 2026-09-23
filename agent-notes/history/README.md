# Machine-readable history

`state.json` is the latest structured handoff state. Rewrite it at each mandatory checkpoint.

`events.jsonl` is append only. Each non-empty line must validate against `event.schema.json`. Use paths to durable evidence instead of copying large logs into an event.

When correcting history, append a new event and set `supersedes` to the earlier `event_id`; never mutate old lines.
