"""Post lead assignments to an n8n webhook (runs alongside Resend/SMTP email)."""

from __future__ import annotations

import os
from typing import Any

import httpx
import pandas as pd

from .features import _safe_str
from .integrations.hubspot import hubspot_owner_id_for_rep, hubspot_owner_resolve_note
from .lead_form_fields import form_entries_for_row, resolve_form_config_for_row
from .phone_utils import format_us_display, format_us_e164, rep_first_name
from .routing_notify import build_assignment_email


def n8n_webhook_url() -> str:
    return os.getenv("N8N_WEBHOOK_URL", "").strip()


def n8n_webhook_secret() -> str:
    return os.getenv("N8N_WEBHOOK_SECRET", "").strip()


def n8n_configured() -> bool:
    return bool(n8n_webhook_url())


def _lead_payload(
    row: pd.Series | dict[str, Any],
    *,
    form_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    get = row.get if isinstance(row, dict) else row.get
    first = _safe_str(get("First Name", ""))
    last = _safe_str(get("Last Name", ""))
    resolved = resolve_form_config_for_row(row, form_config)
    raw_phone = _safe_str(get("Phone Number", ""))
    return {
        "record_id": _safe_str(get("Record ID", "")),
        "first_name": first,
        "last_name": last,
        "name": f"{first} {last}".strip() or _safe_str(get("Email", "")),
        "email": _safe_str(get("Email", "")),
        "phone": raw_phone,
        "phone_display": format_us_display(raw_phone),
        "phone_e164": format_us_e164(raw_phone),
        "state": _safe_str(get("State/Region", "")),
        "message": _safe_str(get("Message", "")),
        "job_title": _safe_str(get("Job Title", "")),
        "job_function": _safe_str(get("Job function", "")),
        "relationship_status": _safe_str(get("Relationship Status", "")),
        "wrestler_goal": _safe_str(get("Wrestler's Goal", "")),
        "wrestler_grade": _safe_str(get("Wrestler's Grade", "")),
        "years_experience": _safe_str(get("Years experience", "")),
        "deadline_for_goal": _safe_str(get("Deadline for Goal", "")),
        "investment_level": _safe_str(get("Investment Level", "")),
        "ai_score": _safe_str(get("AI Score", "")),
        "ai_tier": _safe_str(get("AI Tier", "")),
        "recommended_action": _safe_str(get("Recommended Action", "")),
        "create_date": _safe_str(get("Create Date", "")),
        "source": _safe_str(get("Source", "")),
        "utm_source": _safe_str(get("UTM Source", "")),
        "utm_medium": _safe_str(get("UTM Medium", "")),
        "utm_campaign": _safe_str(get("UTM Campaign", "")),
        "utm_term": _safe_str(get("UTM Term", "")),
        "utm_content": _safe_str(get("UTM Content", "")),
        "utm_keyword": _safe_str(get("UTM Keyword", "")),
        "form": {
            label: val for label, val in form_entries_for_row(row, form_config=resolved)
        },
    }


def _rep_payload_for_n8n(rep: dict[str, Any]) -> dict[str, Any]:
    """Rep block for n8n — owner id, SMS phone, and display name for lead texts."""
    raw_phone = _safe_str(rep.get("phone", ""))
    full_name = _safe_str(rep.get("name", ""))
    payload: dict[str, Any] = {
        "id": _safe_str(rep.get("id", "")),
        "name": full_name,
        "first_name": rep_first_name(full_name),
        "email": _safe_str(rep.get("email", "")),
        "phone": raw_phone,
        "phone_display": format_us_display(raw_phone),
        "phone_e164": format_us_e164(raw_phone),
        "bucket": _safe_str(rep.get("bucket", "")),
        "hubspot_owner_id": None,
        "has_hubspot_owner_id": False,
    }
    owner_id = hubspot_owner_id_for_rep(rep)
    if owner_id.isdigit():
        payload["hubspot_owner_id"] = int(owner_id)
        payload["has_hubspot_owner_id"] = True
    payload["hubspot_owner_note"] = hubspot_owner_resolve_note(rep)
    return payload


def _sms_payload(lead: dict[str, Any], rep: dict[str, Any]) -> dict[str, Any]:
    lead_name = _safe_str(lead.get("first_name")) or "there"
    lead_phone = _safe_str(lead.get("phone_display")) or _safe_str(lead.get("phone"))
    rep_first = _safe_str(rep.get("first_name")) or "Your coach"
    rep_phone = _safe_str(rep.get("phone_display")) or _safe_str(rep.get("phone"))

    rep_lines = [
        "New lead assigned!",
        f"Name: {_safe_str(lead.get('name')) or lead_name}",
        f"Email: {_safe_str(lead.get('email'))}",
    ]
    if lead_phone:
        rep_lines.append(f"Phone: {lead_phone}")
    brief_first = _safe_str(lead.get("brief_first_line"))
    if brief_first:
        rep_lines.append(f"Brief: {brief_first}")
    rep_lines.append("Please follow up ASAP.")

    lead_lines = [
        f"Hi {lead_name}, we received your form.",
    ]
    if rep_phone:
        lead_lines.append(f"{rep_first} will call you from {rep_phone}.")
    else:
        lead_lines.append(f"{rep_first} will contact you soon.")
    lead_lines.append("- Wrestling Mindset")

    return {
        "rep_message": "\n".join(rep_lines),
        "lead_message": "\n".join(lead_lines),
        "rep_to": _safe_str(rep.get("phone_e164")),
        "lead_to": _safe_str(lead.get("phone_e164")),
    }


def build_n8n_payload(
    row: pd.Series | dict[str, Any],
    rep: dict[str, Any],
    assignment: dict[str, Any],
    *,
    test: bool = False,
    form_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    resolved_form = resolve_form_config_for_row(row, form_config)
    subject, text, html = build_assignment_email(
        row, rep, assignment, form_config=resolved_form
    )
    routing = (resolved_form or form_config or {}).get("routing") or {}
    lead_block = _lead_payload(row, form_config=resolved_form)
    from .enrichment import ENRICHMENT_FIELDS
    from .routing_notify import brief_lines_for_row

    get = row.get if isinstance(row, dict) else row.get
    enrichment_block = {key: _safe_str(get(col, "")) for key, col in ENRICHMENT_FIELDS}
    brief_lines = brief_lines_for_row(row)
    enrichment_block["brief_text"] = "\n".join(brief_lines)
    is_error = _safe_str(get("Enrichment Confidence", "")) == "Error"
    lead_block["brief_first_line"] = brief_lines[0] if brief_lines and not is_error else ""
    rep_block = _rep_payload_for_n8n(rep)
    payload: dict[str, Any] = {
        "event": "lead_assignment_test" if test else "lead_assigned",
        "test": test,
        "lead": lead_block,
        "rep": rep_block,
        "sms": _sms_payload(lead_block, rep_block),
        "enrichment": enrichment_block,
        "assignment": {
            "route_bucket": assignment.get("route_bucket", ""),
            "route_reason": assignment.get("route_reason", ""),
            "form_id": assignment.get("form_id") or _safe_str((resolved_form or {}).get("id")),
        },
        "email": {
            "subject": subject,
            "text": text,
            "html": html,
            "to": _safe_str(rep.get("email", "")),
            "format": "html",
        },
    }
    if resolved_form:
        payload["form"] = {
            "id": _safe_str(resolved_form.get("id")),
            "label": _safe_str(resolved_form.get("label")),
            "routing_policy": _safe_str(routing.get("policy", "ai")),
            "wufoo_name": _safe_str(resolved_form.get("wufoo_name")),
        }
    return payload


def _post_to_n8n(payload: dict[str, Any]) -> dict[str, Any]:
    url = n8n_webhook_url()
    if not url:
        raise RuntimeError("N8N_WEBHOOK_URL is not set.")

    headers = {"Content-Type": "application/json"}
    secret = n8n_webhook_secret()
    if secret:
        headers["Authorization"] = secret
        headers["X-Webhook-Secret"] = secret

    with httpx.Client(timeout=30.0) as client:
        response = client.post(url, json=payload, headers=headers)

    if response.status_code >= 400:
        detail = response.text[:300]
        raise RuntimeError(f"n8n webhook returned {response.status_code}: {detail}")

    body: Any = None
    if response.text.strip():
        try:
            body = response.json()
        except Exception:
            body = response.text[:500]

    return {"status_code": response.status_code, "body": body}


def send_n8n_assignment_notification(
    row: pd.Series | dict[str, Any],
    rep: dict[str, Any],
    assignment: dict[str, Any],
    *,
    form_config: dict[str, Any] | None = None,
) -> bool:
    payload = build_n8n_payload(row, rep, assignment, test=False, form_config=form_config)
    _post_to_n8n(payload)
    return True


def verify_n8n_connection() -> dict[str, Any]:
    if not n8n_configured():
        return {
            "ok": False,
            "error": "N8N_WEBHOOK_URL is not set on the server.",
            "transport": "n8n",
        }

    sample_row = {
        "First Name": "Test",
        "Last Name": "Lead",
        "Email": "test@example.com",
        "Phone Number": "555-0100",
        "Message": "This is a test from LeadsWrestling Setup.",
        "Job Title": "Parent Seeking 1-1 Coaching for Child",
        "Relationship Status": "Ready to start now",
    }
    from .routing_config import load_routing_config

    config = load_routing_config()
    sample_rep = next(
        (r for r in config.get("reps", []) if _safe_str(r.get("email")) and r.get("bucket") != "automation"),
        {
            "id": "test-rep",
            "name": "Test Rep",
            "email": "rep@example.com",
            "bucket": "general",
        },
    )
    sample_assignment = {
        "route_bucket": "general",
        "route_reason": "LeadsWrestling n8n connection test",
    }

    test_payload = build_n8n_payload(sample_row, sample_rep, sample_assignment, test=True)
    try:
        result = _post_to_n8n(test_payload)
    except Exception as exc:
        return {"ok": False, "error": str(exc), "transport": "n8n"}

    rep_block = test_payload.get("rep", {})
    return {
        "ok": True,
        "transport": "n8n",
        "status_code": result.get("status_code"),
        "hubspot_owner_id": rep_block.get("hubspot_owner_id"),
        "note": rep_block.get("hubspot_owner_note", "Check n8n → Webhook → body.rep."),
    }


def notify_configured() -> bool:
    from .routing_notify import email_configured

    return email_configured() or n8n_configured()
