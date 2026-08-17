"""Send routed leads to Clay for enrichment; receive results and forward a pre-call brief via n8n."""

from __future__ import annotations

import os
from datetime import UTC, datetime
from html import escape
from typing import Any

import httpx
import pandas as pd

from .features import _safe_str
from .phone_utils import format_us_e164

ENRICHMENT_FIELDS: tuple[tuple[str, str], ...] = (
    # (Clay callback key, cache column)
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
ENRICHMENT_COLUMNS: tuple[str, ...] = tuple(col for _, col in ENRICHMENT_FIELDS) + ("Enriched At",)


def clay_webhook_url() -> str:
    return os.getenv("CLAY_WEBHOOK_URL", "").strip()


def clay_configured() -> bool:
    return bool(clay_webhook_url())


def clay_callback_secret() -> str:
    return os.getenv("CLAY_CALLBACK_SECRET", "").strip()


def enrichment_n8n_url() -> str:
    return os.getenv("N8N_ENRICHMENT_WEBHOOK_URL", "").strip()


def _public_api_url() -> str:
    return os.getenv("PUBLIC_API_URL", "").strip() or "https://wrestlingleads-production.up.railway.app"


def build_clay_payload(
    row: pd.Series | dict[str, Any],
    rep: dict[str, Any],
    assignment: dict[str, Any],
    *,
    form_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Flat payload — Clay builds one table column per top-level key."""
    get = row.get if isinstance(row, dict) else row.get
    first = _safe_str(get("First Name", ""))
    last = _safe_str(get("Last Name", ""))
    email = _safe_str(get("Email", ""))
    return {
        "lead_id": _safe_str(get("Record ID", "")) or email,
        "email": email,
        "form_name": f"{first} {last}".strip(),
        "first_name": first,
        "last_name": last,
        "buyer_type": _safe_str(get("Job Title", "")),
        "state": _safe_str(get("State/Region", "")),
        "grade": _safe_str(get("Wrestler's Grade", "")),
        "years_experience": _safe_str(get("Years experience", "")),
        "wrestler_goal": _safe_str(get("Wrestler's Goal", "")),
        "reason_for_inquiry": _safe_str(get("Job function", "")),
        "message": _safe_str(get("Message", "")),
        "assigned_rep": _safe_str(rep.get("name", "")),
        "rep_id": _safe_str(rep.get("id", "")),
        "rep_email": _safe_str(rep.get("email", "")),
        "route_bucket": _safe_str(assignment.get("route_bucket", "")),
        "ai_tier": _safe_str(get("AI Tier", "")),
        "ai_score": _safe_str(get("AI Score", "")),
        "form_id": _safe_str((form_config or {}).get("id")),
        "callback_url": f"{_public_api_url().rstrip('/')}/webhooks/clay-enrichment",
    }


def send_clay_enrichment_request(
    row: pd.Series | dict[str, Any],
    rep: dict[str, Any],
    assignment: dict[str, Any],
    *,
    form_config: dict[str, Any] | None = None,
) -> bool:
    url = clay_webhook_url()
    if not url:
        raise RuntimeError("CLAY_WEBHOOK_URL is not set.")
    payload = build_clay_payload(row, rep, assignment, form_config=form_config)
    with httpx.Client(timeout=20.0) as client:
        response = client.post(url, json=payload, headers={"Content-Type": "application/json"})
    if response.status_code >= 400:
        raise RuntimeError(f"Clay webhook returned {response.status_code}: {response.text[:300]}")
    return True


def parse_enrichment_body(body: dict[str, Any]) -> dict[str, str]:
    """Map Clay's callback JSON onto cache columns (missing keys → empty)."""
    values = {col: _safe_str(body.get(key, "")) for key, col in ENRICHMENT_FIELDS}
    values["Enriched At"] = datetime.now(UTC).strftime("%Y-%m-%d %H:%M")
    return values


def _brief_lines(lead: pd.Series | dict[str, Any], enrichment: dict[str, str]) -> list[str]:
    get = lead.get if isinstance(lead, dict) else lead.get
    lines: list[str] = []
    wrestler = enrichment.get("Wrestler Name") or "Wrestler (name not on form)"
    detail = " · ".join(
        p
        for p in (
            _safe_str(get("Wrestler's Grade", "")),
            _safe_str(get("Years experience", "")),
        )
        if p
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
        + (f"<p>{escape(summary)}</p>" if summary else "")
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
