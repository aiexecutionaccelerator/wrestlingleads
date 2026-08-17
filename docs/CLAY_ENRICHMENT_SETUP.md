# Lead Enrichment — Setup

Two providers, selected by `ENRICHMENT_PROVIDER` (`claude` | `clay` | `off`; default: `claude` if `ANTHROPIC_API_KEY` is set, else `clay` if `CLAY_WEBHOOK_URL` is set, else off). Automation-bucket leads are never enriched.

## Provider A (recommended): Claude API — no Clay account needed

`backend/app/enrichment.py`. After routing, a background thread makes one Claude call (`claude-opus-5`, web search + web fetch server tools) that: works out parent vs wrestler from the form, searches TrackWrestling for the record / weight / team, looks up the club, and writes a 3-line pre-call brief. Result is stored on the lead (lead card) and forwarded to `N8N_ENRICHMENT_WEBHOOK_URL` for the rep's email/SMS.

Railway env vars:

| Var | Value |
|---|---|
| `ANTHROPIC_API_KEY` | from console.anthropic.com |
| `N8N_ENRICHMENT_WEBHOOK_URL` | production URL of the n8n workflow imported from `docs/n8n_enrichment_workflow.json` |
| `CLAY_CALLBACK_SECRET` | any long string — also protects the manual `/webhooks/enrichment/run` endpoint |
| `ENRICHMENT_MODEL` (optional) | default `claude-opus-5`; `claude-sonnet-5` is cheaper |
| `ENRICHMENT_MAX_SEARCHES` / `ENRICHMENT_MAX_FETCHES` (optional) | default 8 each — cost guardrail |

Cost: web search is $10 per 1,000 searches plus tokens — roughly $0.15–0.25 per lead on Opus.

Check: `GET /webhooks/enrichment/status`. Manually enrich an existing lead: `POST /webhooks/enrichment/run?email=<lead email>` with header `X-Clay-Secret: <CLAY_CALLBACK_SECRET>` (returns the brief; add `&wait=false` to run in the background).

## Provider B: Clay

Flow: Wufoo → backend (score + route) → **Clay** (webhook table, AI enrichment) → backend callback → n8n (email + SMS brief to rep) and the lead card in the app. Set `ENRICHMENT_PROVIDER=clay` (or leave `ANTHROPIC_API_KEY` unset).

## Railway env vars

| Var | Value |
|---|---|
| `CLAY_WEBHOOK_URL` | Clay table → "Import from webhook" URL |
| `CLAY_CALLBACK_SECRET` | any long random string; Clay sends it back as header `X-Clay-Secret` |
| `N8N_ENRICHMENT_WEBHOOK_URL` | production URL of the n8n workflow imported from `docs/n8n_enrichment_workflow.json` |
| `PUBLIC_API_URL` | (already set) used to build `callback_url` in the Clay payload |

Check with `GET /webhooks/clay/status`.

## Payload the backend sends to Clay (one column per key)

`lead_id, email, form_name, first_name, last_name, buyer_type, state, grade, years_experience, wrestler_goal, reason_for_inquiry, message, assigned_rep, rep_id, rep_email, route_bucket, ai_tier, ai_score, form_id, callback_url`

## Clay table columns

All Claygent columns: **structured output ON**, set "Only run if" as noted.

**0. Who's who** (cheapest model)
> The form's Name field is "{{form_name}}", buyer type is "{{buyer_type}}", and the free-text message is: "{{message}}". Determine whether the Name field is the parent or the wrestler. Rules: if buyer type is "Wrestler Seeking…" the name is the wrestler. If buyer type is "Parent…", the name is the parent *unless* the message clearly refers to that same first name as the athlete (e.g. "Jessie wrestled…" with Name "Jessie Murphy" → wrestler). Extract any other name the message gives for the child. Return `parent_name`, `wrestler_name` (empty if unknown), `confidence` (High/Medium/Low), `reasoning` (one sentence).

**1. TrackWrestling record** — only if `wrestler_name` not empty; mid-tier model
> Search TrackWrestling.com (fallback: FloWrestling, school athletics pages) for a youth/middle-school wrestler named **{{wrestler_name}}** from **{{state}}**, grade **{{grade}}**. Find the most recent season's win-loss record, weight class, and team/school. Only report a match if name AND state both match — never guess between same-name wrestlers. Return `found`, `record` (e.g. "23-6"), `weight_class`, `team`, `season`, `source_url`, `confidence` (High/Medium/Low), `notes`.

**2. Club affiliation** — only if `wrestler_name` not empty
> Identify the wrestling club/team for **{{wrestler_name}}**, a **{{grade}}** wrestler in **{{state}}**. Check USA Wrestling club listings, TrackWrestling team pages, club rosters. Return `club_name`, `club_city`, `source_url`, `confidence`, or `found: false`. Do not report a club unless the wrestler's name appears on that club's roster or results.

**3. Parent LinkedIn** — only if `parent_name` not empty AND `buyer_type` contains "Parent". Use Clay's native *Find LinkedIn profile from name* (name + state), then a small Claygent column for headline/company.

**4. Confidence rollup** (formula) — High if TW=High; Medium if TW or club found; Low if only LinkedIn; else "Not found".

**5. Summary for Jake** (cheapest model)
> Write a 3-line pre-call brief for a sales rep. Line 1: wrestler name, grade, experience, record and team if found (say "no record found" if not; if the wrestler's name is unknown say so). Line 2: club if found. Line 3: parent's role/company from LinkedIn if found, otherwise the parent's own words from the form message. Factual, no fluff; mark Low-confidence facts as "possibly."

**6. Send to backend** — column type **HTTP API**, only if Summary is filled
- Method: `POST`
- URL: `{{callback_url}}`
- Headers: `Content-Type: application/json`, `X-Clay-Secret: <CLAY_CALLBACK_SECRET>`
- Body:
```json
{
  "lead_id": "{{lead_id}}",
  "email": "{{email}}",
  "rep_id": "{{rep_id}}",
  "wrestler_name": "{{wrestler_name}}",
  "parent_name": "{{parent_name}}",
  "tw_record": "{{record}}",
  "tw_weight_class": "{{weight_class}}",
  "tw_team": "{{team}}",
  "tw_source_url": "{{source_url}}",
  "club": "{{club_name}}",
  "parent_linkedin_url": "{{linkedin_url}}",
  "parent_headline": "{{headline}}",
  "confidence": "{{confidence_rollup}}",
  "summary": "{{summary}}"
}
```
Expect `200 {"success": true, ...}`. `404` means the email isn't in the app's cache (e.g. a Clay test row that never came through Wufoo).

Set every column to auto-run on new rows.

## n8n

Import `docs/n8n_enrichment_workflow.json` (uses the same Gmail/Twilio credentials as the assignment workflow). Activate it, copy the **production** webhook URL into `N8N_ENRICHMENT_WEBHOOK_URL`.

Payload it receives: `body.rep.{name,email,phone_e164}`, `body.lead.{email,name}`, `body.email.{subject,html,text}`, `body.sms.{rep_to,rep_message}`, `body.note` (plain-text brief), `body.enrichment.{...}`.

Optional HubSpot step (needs a HubSpot login only for the OAuth already stored in n8n): add **HubSpot → Contact → Search** (email = `{{ $json.body.lead.email }}`) → **HubSpot → Engagement → Create** (type Note, body `{{ $('Enrichment Webhook').item.json.body.note }}`, associate to the found contact ID). Notes need no custom properties.

## Later (needs Wufoo / HubSpot admin)

- Wufoo: add "Wrestler's First & Last Name" to the 1-on-1 form → map it in `config/wufoo_field_map.json` → pass it as `wrestler_name` in the Clay payload and skip Column 0.
- HubSpot: create `lw_*` enrichment properties and add a Clay HubSpot "Create or Update Contact" column (match on email).
