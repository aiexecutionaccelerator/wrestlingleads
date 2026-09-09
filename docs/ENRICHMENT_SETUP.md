# Lead Enrichment — Setup

When a lead is scored and routed, `backend/app/enrichment.py` makes one Claude call (`claude-opus-5` with the web search + web fetch server tools) **before** the assignment notification goes out. It works out parent vs wrestler from the form, searches TrackWrestling for the wrestler's record / weight / team, looks up their club, and writes a 3-line pre-call brief.

The brief is then part of the **one** assignment notification the rep already gets:
- **Email** (`body.email.html` / `.text` sent by the existing n8n Gmail node) — a highlighted "Pre-call brief" box above the form fields.
- **Rep SMS** (`body.sms.rep_message`) — first line of the brief.
- **n8n payload** — new `body.enrichment.{wrestler_name, parent_name, tw_record, tw_weight_class, tw_team, tw_source_url, club, confidence, summary, brief_text}` for optional HubSpot mapping.
- **Lead card** in the app — "Pre-call brief".

**The existing n8n workflow needs no changes** — the backend builds the email body it sends. To also write the brief to HubSpot, map `body.enrichment.brief_text` to a property (or a Note) in the existing HubSpot node.

Trade-off: the notification waits for the research (typically 30–120 s), bounded by `ENRICHMENT_TIMEOUT_SECONDS`. If research fails or times out, the notification goes out immediately without a brief. Automation-bucket (nurture) leads are never enriched.

## Railway env vars

| Var | Value |
|---|---|
| `ANTHROPIC_API_KEY` | from console.anthropic.com → API Keys. Enrichment is on whenever this is set. |
| `ENRICHMENT_SECRET` | any long string — protects the manual `/webhooks/enrichment/run` endpoint (legacy name `CLAY_CALLBACK_SECRET` still works) |
| `ENRICHMENT_PROVIDER` (optional) | set to `off` to disable without removing the key |
| `ENRICHMENT_MODEL` (optional) | default `claude-opus-5`; `claude-sonnet-5` is cheaper |
| `ENRICHMENT_TIMEOUT_SECONDS` (optional) | default 150 — per-call cap before the notification goes out without a brief |
| `ENRICHMENT_MAX_SEARCHES` / `ENRICHMENT_MAX_FETCHES` (optional) | defaults 5 / 4 — cost guardrail |
| `ENRICHMENT_FETCH_MAX_TOKENS` (optional) | default 15000 — caps how much of each fetched web page is ingested |

Cost: roughly $0.50–0.90 per lead on Opus (page-size caps + prompt caching; ~40% less on `claude-sonnet-5`). Each run logs `est_cost=$…` in Railway logs.

## Checking it

- `GET /webhooks/enrichment/status` — enabled?, model, timeout.
- `POST /webhooks/enrichment/run?email=<lead email>` with header `X-Enrichment-Secret: <ENRICHMENT_SECRET>` — research an existing lead now, store the brief on it, and return it. Does not re-send notifications.

## Later (needs Wufoo / HubSpot admin)

- Wufoo: add "Wrestler's First & Last Name" to the 1-on-1 form → map it in `config/wufoo_field_map.json` → include it in `_lead_description()` so the model no longer has to infer it.
- HubSpot: create `lw_*` enrichment properties and map `body.enrichment.*` to them in the existing n8n HubSpot node.
