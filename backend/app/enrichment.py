"""Lead enrichment with Claude + web search: wrestler record, club, and a pre-call brief for the rep.

Enabled when ANTHROPIC_API_KEY is set (disable with ENRICHMENT_PROVIDER=off).
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
from datetime import UTC, datetime
from html import escape
from typing import Any

import httpx
import pandas as pd

from .features import _safe_str
from .phone_utils import format_us_e164

logger = logging.getLogger(__name__)

ENRICHMENT_MODEL = os.getenv("ENRICHMENT_MODEL", "claude-opus-5")
MAX_SEARCHES = int(os.getenv("ENRICHMENT_MAX_SEARCHES", "8"))
MAX_FETCHES = int(os.getenv("ENRICHMENT_MAX_FETCHES", "8"))

ENRICHMENT_FIELDS: tuple[tuple[str, str], ...] = (
    # (result key, cache column)
    ("wrestler_name", "Wrestler Name"),
    ("parent_name", "Parent Name"),
    ("tw_record", "TW Record"),
    ("tw_weight_class", "TW Weight Class"),
    ("tw_team", "TW Team"),
    ("tw_source_url", "TW Source URL"),
    ("club", "Club Affiliation"),
    ("parent_linkedin_url", "Parent LinkedIn"),
    ("parent_headline", "Parent Headline"),
    ("confidence", "Enrichment Confidence"),
    ("summary", "Enrichment Summary"),
)
RESULT_KEYS = tuple(key for key, _ in ENRICHMENT_FIELDS)
ENRICHMENT_COLUMNS: tuple[str, ...] = tuple(col for _, col in ENRICHMENT_FIELDS) + ("Enriched At",)

SYSTEM_PROMPT = """You research inbound leads for Wrestling Mindset, a 1-on-1 mental-performance coaching program for youth and high-school wrestlers, so the assigned sales rep is prepared before their first call.

You receive one lead's form submission. Do the following, using web search and page fetches as needed:

1. Determine who is who. The form's Name field is sometimes the parent and sometimes the wrestler.
   - Buyer type "Wrestler Seeking..." → the name is the wrestler.
   - Buyer type "Parent..." → the name is the parent, unless the message clearly uses that same first name for the athlete (e.g. Name "Jessie Murphy" + message "Jessie wrestled for one year..." → Jessie is the wrestler). Also extract any child's name given in the message. If the wrestler's name cannot be determined, leave wrestler_name empty — do not guess.
2. If you have a wrestler name: search TrackWrestling.com (fallback: FloWrestling, school athletics pages) for that wrestler in the given state and grade level. Report the most recent season's win-loss record, weight class, team/school, and the source URL. Only report a match if name AND state agree — never guess between same-name wrestlers. Youth and middle-school coverage is thin; "not found" is a normal outcome.
3. If you have a wrestler name: identify their club/team via USA Wrestling club listings, TrackWrestling team pages, or club rosters. Report a club only if the wrestler's name appears on that club's roster or results.
4. Write a 3-line pre-call brief for the rep, plain text, no markdown:
   Line 1: wrestler name (or "wrestler name not on form — ask on call"), grade, experience, record/weight/team if found, otherwise "no TrackWrestling record found".
   Line 2: club if found; otherwise the parent's own words from the form message.
   Line 3: reason for inquiry and goal, plus anything from your research that changes how the rep should open the call.
   Be factual. Mark low-confidence facts as "possibly". Never state something as fact that you did not verify on a page.

Budget: use at most a handful of searches. Stop as soon as you have an answer or it is clear nothing reliable exists.

Return ONLY a JSON object with exactly these keys (use "" for unknown):
wrestler_name, parent_name, tw_record, tw_weight_class, tw_team, tw_source_url, club, parent_linkedin_url, parent_headline, confidence, summary
- confidence: one of "High", "Medium", "Low", "Not found" (High = record verified on TrackWrestling with matching state; Medium = record or club found with minor ambiguity; Low = only weak signals; Not found = nothing verified).
- summary: the 3-line brief, lines separated by "\\n".
- parent_linkedin_url / parent_headline: only if you happened to find a clearly matching public profile; otherwise "".
"""


# ---------------------------------------------------------------- config


def enrichment_secret() -> str:
    return os.getenv("ENRICHMENT_SECRET", "").strip() or os.getenv("CLAY_CALLBACK_SECRET", "").strip()


def enrichment_n8n_url() -> str:
    return os.getenv("N8N_ENRICHMENT_WEBHOOK_URL", "").strip()


def enrichment_enabled() -> bool:
    if os.getenv("ENRICHMENT_PROVIDER", "").strip().lower() == "off":
        return False
    return bool(os.getenv("ANTHROPIC_API_KEY", "").strip())


# ---------------------------------------------------------------- Claude research


def _lead_description(row: pd.Series | dict[str, Any], rep: dict[str, Any] | None) -> str:
    get = row.get if isinstance(row, dict) else row.get
    first = _safe_str(get("First Name", ""))
    last = _safe_str(get("Last Name", ""))
    fields = [
        ("Name on form", f"{first} {last}".strip()),
        ("Buyer type", _safe_str(get("Job Title", ""))),
        ("State", _safe_str(get("State/Region", ""))),
        ("Wrestler's grade", _safe_str(get("Wrestler's Grade", ""))),
        ("Years of experience", _safe_str(get("Years experience", ""))),
        ("Wrestler's goal", _safe_str(get("Wrestler's Goal", ""))),
        ("Reason for inquiry", _safe_str(get("Job function", ""))),
        ("Deadline", _safe_str(get("Deadline for Goal", ""))),
        ("Investment level", _safe_str(get("Investment Level", ""))),
        ("Message", _safe_str(get("Message", ""))),
        ("Assigned rep", _safe_str((rep or {}).get("name", ""))),
    ]
    lines = [f"{label}: {value}" for label, value in fields if value]
    return "Lead form submission:\n" + "\n".join(lines)


def _extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object in enrichment response: {text[:200]}")
    return json.loads(match.group(0))


def _tools() -> list[dict[str, Any]]:
    return [
        {"type": "web_search_20260209", "name": "web_search", "max_uses": MAX_SEARCHES},
        {"type": "web_fetch_20260209", "name": "web_fetch", "max_uses": MAX_FETCHES},
    ]


def research_lead_with_claude(
    row: pd.Series | dict[str, Any],
    rep: dict[str, Any] | None = None,
    *,
    client: Any | None = None,
) -> dict[str, str]:
    """One Claude call with web search/fetch; returns a dict keyed by RESULT_KEYS."""
    if client is None:
        import anthropic

        client = anthropic.Anthropic()

    messages: list[dict[str, Any]] = [{"role": "user", "content": _lead_description(row, rep)}]
    response = None
    for _ in range(4):  # pause_turn continuations
        response = client.messages.create(
            model=ENRICHMENT_MODEL,
            max_tokens=8000,
            system=SYSTEM_PROMPT,
            tools=_tools(),
            messages=messages,
        )
        if response.stop_reason != "pause_turn":
            break
        messages = [messages[0], {"role": "assistant", "content": response.content}]

    if response is None:
        raise RuntimeError("Enrichment produced no response.")
    if response.stop_reason == "refusal":
        raise RuntimeError("Enrichment request was declined by the model.")

    text = "".join(block.text for block in response.content if getattr(block, "type", "") == "text")
    parsed = _extract_json(text)
    result = {key: _safe_str(parsed.get(key, "")) for key in RESULT_KEYS}

    usage = getattr(response, "usage", None)
    server_use = getattr(usage, "server_tool_use", None)
    get = row.get if isinstance(row, dict) else row.get
    logger.info(
        "Enrichment done email=%s confidence=%s searches=%s in=%s out=%s",
        _safe_str(get("Email", "")),
        result.get("confidence"),
        getattr(server_use, "web_search_requests", None),
        getattr(usage, "input_tokens", None),
        getattr(usage, "output_tokens", None),
    )
    return result


# ---------------------------------------------------------------- store + notify


def result_to_columns(result: dict[str, Any]) -> dict[str, str]:
    values = {col: _safe_str(result.get(key, "")) for key, col in ENRICHMENT_FIELDS}
    values["Enriched At"] = datetime.now(UTC).strftime("%Y-%m-%d %H:%M")
    return values


def _brief_lines(lead: pd.Series | dict[str, Any], enrichment: dict[str, str]) -> list[str]:
    get = lead.get if isinstance(lead, dict) else lead.get
    lines: list[str] = []
    wrestler = enrichment.get("Wrestler Name") or "Wrestler (name not on form)"
    detail = " · ".join(
        p for p in (_safe_str(get("Wrestler's Grade", "")), _safe_str(get("Years experience", ""))) if p
    )
    lines.append(f"{wrestler}" + (f" — {detail}" if detail else ""))
    record = enrichment.get("TW Record")
    if record:
        rec = f"TrackWrestling: {record}"
        if enrichment.get("TW Weight Class"):
            rec += f" @ {enrichment['TW Weight Class']}"
        if enrichment.get("TW Team"):
            rec += f" ({enrichment['TW Team']})"
        lines.append(rec)
    else:
        lines.append("TrackWrestling: no record found")
    if enrichment.get("Club Affiliation"):
        lines.append(f"Club: {enrichment['Club Affiliation']}")
    if enrichment.get("Parent LinkedIn"):
        parent = enrichment.get("Parent Name") or "Parent"
        headline = enrichment.get("Parent Headline")
        lines.append(f"{parent}: {headline + ' — ' if headline else ''}{enrichment['Parent LinkedIn']}")
    if enrichment.get("Enrichment Confidence"):
        lines.append(f"Confidence: {enrichment['Enrichment Confidence']}")
    return lines


def build_enrichment_n8n_payload(
    lead: pd.Series | dict[str, Any],
    rep: dict[str, Any] | None,
    enrichment: dict[str, str],
) -> dict[str, Any]:
    get = lead.get if isinstance(lead, dict) else lead.get
    first = _safe_str(get("First Name", ""))
    last = _safe_str(get("Last Name", ""))
    name = f"{first} {last}".strip() or _safe_str(get("Email", ""))
    rep = rep or {}
    rep_phone = _safe_str(rep.get("phone", ""))

    lines = _brief_lines(lead, enrichment)
    summary = enrichment.get("Enrichment Summary", "")
    subject = f"Pre-call brief: {name}"
    text_parts = [f"Pre-call brief for {name}", ""] + lines
    if summary:
        text_parts += ["", summary]
    if enrichment.get("TW Source URL"):
        text_parts += ["", f"Source: {enrichment['TW Source URL']}"]
    text = "\n".join(text_parts)

    html_items = "".join(f"<li>{escape(line)}</li>" for line in lines)
    html = (
        f"<p>Pre-call brief for <strong>{escape(name)}</strong></p>"
        f"<ul>{html_items}</ul>"
        + (f"<p>{escape(summary).replace(chr(10), '<br>')}</p>" if summary else "")
        + (
            f'<p><a href="{escape(enrichment["TW Source URL"])}">TrackWrestling source</a></p>'
            if enrichment.get("TW Source URL")
            else ""
        )
    )

    sms_body = "\n".join([f"Brief: {name}"] + lines[:3])
    if len(sms_body) > 300:
        sms_body = sms_body[:297] + "..."

    return {
        "event": "lead_enriched",
        "lead": {
            "record_id": _safe_str(get("Record ID", "")),
            "email": _safe_str(get("Email", "")),
            "name": name,
            "first_name": first,
            "last_name": last,
        },
        "rep": {
            "id": _safe_str(rep.get("id", "")),
            "name": _safe_str(rep.get("name", "")),
            "email": _safe_str(rep.get("email", "")),
            "phone_e164": format_us_e164(rep_phone),
        },
        "enrichment": {key: enrichment.get(col, "") for key, col in ENRICHMENT_FIELDS},
        "note": text,
        "email": {"subject": subject, "text": text, "html": html, "to": _safe_str(rep.get("email", ""))},
        "sms": {"rep_to": format_us_e164(rep_phone), "rep_message": sms_body},
    }


def send_enrichment_to_n8n(payload: dict[str, Any]) -> bool:
    url = enrichment_n8n_url()
    if not url:
        raise RuntimeError("N8N_ENRICHMENT_WEBHOOK_URL is not set.")
    headers = {"Content-Type": "application/json"}
    secret = os.getenv("N8N_WEBHOOK_SECRET", "").strip()
    if secret:
        headers["Authorization"] = secret
        headers["X-Webhook-Secret"] = secret
    with httpx.Client(timeout=30.0) as client:
        response = client.post(url, json=payload, headers=headers)
    if response.status_code >= 400:
        raise RuntimeError(f"n8n enrichment webhook returned {response.status_code}: {response.text[:300]}")
    return True


def record_enrichment(email: str, result: dict[str, Any], rep_id: str = "") -> dict[str, Any]:
    """Store enrichment values on the lead and forward the pre-call brief to n8n."""
    from .routing_config import get_rep_by_id, load_routing_config
    from .store import store

    values = result_to_columns(result)
    row = store.apply_enrichment(email, values)
    if row is None:
        return {"stored": False, "reason": f"No lead in cache with email {email}."}

    n8n_sent = False
    n8n_error: str | None = None
    if enrichment_n8n_url():
        config = load_routing_config()
        rep = next(
            (r for r in config.get("reps", []) if _safe_str(r.get("email")) == _safe_str(row.get("Assigned Email"))),
            None,
        ) or get_rep_by_id(config, _safe_str(rep_id))
        try:
            n8n_sent = send_enrichment_to_n8n(build_enrichment_n8n_payload(row, rep, values))
        except Exception as exc:
            n8n_error = str(exc)
            logger.warning("Enrichment n8n forward failed for %s: %s", email, exc)

    return {
        "stored": True,
        "stored_fields": [k for k, v in values.items() if v],
        "n8n_sent": n8n_sent,
        "n8n_error": n8n_error,
    }


def enrich_lead(row: pd.Series | dict[str, Any], rep: dict[str, Any] | None) -> dict[str, Any]:
    """Research + store + notify, synchronously (call from a worker thread or a test endpoint)."""
    get = row.get if isinstance(row, dict) else row.get
    email = _safe_str(get("Email", ""))
    result = research_lead_with_claude(row, rep)
    outcome = record_enrichment(email, result, rep_id=_safe_str((rep or {}).get("id", "")))
    return {"result": result, **outcome}


def start_enrichment(row: pd.Series | dict[str, Any], rep: dict[str, Any] | None) -> None:
    """Fire-and-forget: research can take a minute or more, so never block routing on it."""
    snapshot = dict(row)

    def _run() -> None:
        try:
            enrich_lead(snapshot, rep)
        except Exception:
            logger.exception("Enrichment failed for email=%s", snapshot.get("Email"))

    threading.Thread(target=_run, name="lead-enrichment", daemon=True).start()
