# Lead Enrichment — Setup

After a lead is scored and routed, `backend/app/enrichment.py` runs one Claude call in a background thread (`claude-opus-5` with the web search + web fetch server tools). It works out parent vs wrestler from the form, searches TrackWrestling for the wrestler's record / weight / team, looks up their club, and writes a 3-line pre-call brief for the assigned rep. The result is stored on the lead (shown on the lead card as "Pre-call brief") and forwarded to n8n for the rep's email + SMS.

Automation-bucket (nurture) leads are never enriched. Routing never waits on enrichment.

## Railway env vars

| Var | Value |
|---|---|
| `ANTHROPIC_API_KEY` | from console.anthropic.com → API Keys. Enrichment is on whenever this is set. |
| `N8N_ENRICHMENT_WEBHOOK_URL` | production URL of the n8n workflow imported from `docs/n8n_enrichment_workflow.json` |
| `ENRICHMENT_SECRET` | any long string — protects the manual `/webhooks/enrichment/run` endpoint (legacy name `CLAY_CALLBACK_SECRET` still works) |
| `ENRICHMENT_PROVIDER` (optional) | set to `off` to disable without removing the key |
| `ENRICHMENT_MODEL` (optional) | default `claude-opus-5`; `claude-sonnet-5` is cheaper |
| `ENRICHMENT_MAX_SEARCHES` / `ENRICHMENT_MAX_FETCHES` (optional) | default 8 each — cost guardrail |

Cost: web search is $10 per 1,000 searches plus tokens — roughly $0.15–0.25 per lead on Opus.

## Checking it

- `GET /webhooks/enrichment/status` — enabled?, model, n8n configured?
- `POST /webhooks/enrichment/run?email=<lead email>` with header `X-Enrichment-Secret: <ENRICHMENT_SECRET>` — enrich an existing lead now and return the brief. Add `&wait=false` to run in the background instead.

## n8n

Import `docs/n8n_enrichment_workflow.json` (uses the same Gmail/Twilio credentials as the assignment workflow). Activate it, copy the **production** webhook URL into `N8N_ENRICHMENT_WEBHOOK_URL`.

Payload it receives: `body.rep.{name,email,phone_e164}`, `body.lead.{email,name}`, `body.email.{subject,html,text}`, `body.sms.{rep_to,rep_message}`, `body.note` (plain-text brief), `body.enrichment.{wrestler_name, parent_name, tw_record, tw_weight_class, tw_team, tw_source_url, club, confidence, summary, ...}`.

Optional HubSpot step (uses the HubSpot OAuth already stored in n8n; no custom properties needed): **HubSpot → Contact → Search** (email = `{{ $json.body.lead.email }}`) → **HubSpot → Engagement → Create** (type Note, body `{{ $('Enrichment Webhook').item.json.body.note }}`, associate to the found contact ID).

## Later (needs Wufoo / HubSpot admin)

- Wufoo: add "Wrestler's First & Last Name" to the 1-on-1 form → map it in `config/wufoo_field_map.json` → include it in `_lead_description()` so the model no longer has to infer it.
- HubSpot: create `lw_*` enrichment properties and write them from n8n's HubSpot node.
