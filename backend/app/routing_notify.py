"""Email notifications when a lead is routed to a sales rep."""

from __future__ import annotations

import html as html_module
import os
import re
import smtplib
import ssl
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.utils import formataddr, parseaddr
from typing import Any

import httpx
import pandas as pd

from .features import _safe_str
from .lead_form_fields import form_entries_for_row, lead_display_name

RAILWAY_SMTP_HINT = (
    "Railway blocks outbound Gmail SMTP on Free and Hobby plans (Network unreachable). "
    "Fix: upgrade Railway to Pro and redeploy, OR set RESEND_API_KEY to send email over HTTPS instead."
)


def _normalize_app_password(value: str) -> str:
    """Google app passwords are 16 chars — users often paste with spaces."""
    return re.sub(r"\s+", "", value.strip())


def _resend_api_key() -> str:
    return os.getenv("RESEND_API_KEY", "").strip()


def _resend_sandbox_to() -> str:
    """Inbox that receives all rep emails until the domain is verified in Resend."""
    return os.getenv("RESEND_SANDBOX_TO", "").strip()


def resend_sandbox_enabled() -> bool:
    """Resend allows onboarding@resend.dev without DNS — only to your account email."""
    return bool(_resend_api_key() and _resend_sandbox_to())


def _resolve_delivery(rep_email: str) -> tuple[str, str | None]:
    """Return (to_address, optional sandbox banner for the rep)."""
    rep_email = _safe_str(rep_email)
    sandbox_to = _resend_sandbox_to()
    if resend_sandbox_enabled():
        banner = (
            f"⚠️ SANDBOX MODE — This lead was assigned to {rep_email or 'a rep'}. "
            "Verify wrestlingmindset.com in Resend to email reps directly."
        )
        return sandbox_to, banner
    return rep_email, None


def _smtp_settings() -> dict[str, str | int]:
    host = os.getenv("SMTP_HOST", "").strip()
    user = os.getenv("SMTP_USER", "").strip()
    password = _normalize_app_password(os.getenv("SMTP_PASSWORD", ""))
    from_raw = os.getenv("ROUTING_FROM_EMAIL", user).strip()
    _, from_addr = parseaddr(from_raw)
    from_email = from_addr or user
    port = int(os.getenv("SMTP_PORT", "587") or "587")
    return {
        "host": host,
        "user": user,
        "password": password,
        "from_raw": from_raw,
        "from_email": from_email,
        "port": port,
    }


def email_transport() -> str | None:
    if _resend_api_key():
        return "resend"
    if smtp_configured():
        return "smtp"
    return None


def smtp_configured() -> bool:
    settings = _smtp_settings()
    return bool(settings["host"] and settings["user"] and settings["password"])


def email_configured() -> bool:
    return email_transport() is not None


def _from_header() -> str:
    if resend_sandbox_enabled():
        return "Leads Wrestling <onboarding@resend.dev>"
    settings = _smtp_settings()
    from_raw = str(settings["from_raw"])
    from_email = str(settings["from_email"])
    if from_raw and not resend_sandbox_enabled():
        return from_raw
    return formataddr(("Leads Wrestling", from_email))


def _smtp_network_error_hint(exc: Exception) -> str | None:
    msg = str(exc).lower()
    if "101" in msg or "network is unreachable" in msg:
        return RAILWAY_SMTP_HINT
    if "timed out" in msg or "timeout" in msg:
        return (
            f"Could not reach the mail server in time. {RAILWAY_SMTP_HINT}"
        )
    return None


def verify_resend_connection() -> dict[str, Any]:
    api_key = _resend_api_key()
    if not api_key:
        return {"ok": False, "error": "RESEND_API_KEY is not set.", "transport": "resend"}

    try:
        with httpx.Client(timeout=15.0) as client:
            response = client.get(
                "https://api.resend.com/domains",
                headers={"Authorization": f"Bearer {api_key}"},
            )
    except Exception as exc:
        return {"ok": False, "error": str(exc), "transport": "resend"}

    if response.status_code == 401:
        return {
            "ok": False,
            "error": "Invalid RESEND_API_KEY.",
            "transport": "resend",
        }
    if response.status_code >= 400:
        detail = response.text[:200]
        return {
            "ok": False,
            "error": f"Resend API error ({response.status_code}): {detail}",
            "transport": "resend",
        }

    data = response.json()
    domains = data.get("data") or []
    verified = [d for d in domains if d.get("status") == "verified"]
    result: dict[str, Any] = {
        "ok": True,
        "transport": "resend",
        "domains_verified": len(verified),
        "from": _from_header(),
    }
    if resend_sandbox_enabled():
        result["sandbox"] = True
        result["sandbox_to"] = _resend_sandbox_to()
        result["note"] = (
            "Sending via Resend sandbox (no DNS). All rep emails go to RESEND_SANDBOX_TO until "
            "wrestlingmindset.com is verified."
        )
    elif not verified:
        result["note"] = (
            "API key works but no verified domain yet. Set RESEND_SANDBOX_TO to your Resend "
            "signup email for a no-DNS workaround."
        )
    return result


def verify_smtp_connection() -> dict[str, Any]:
    """Try SMTP login only — used for diagnostics."""
    settings = _smtp_settings()
    host = str(settings["host"])
    user = str(settings["user"])
    password = str(settings["password"])
    port = int(settings["port"])

    if not all([host, user, password]):
        return {
            "ok": False,
            "error": "SMTP not configured. Set SMTP_HOST, SMTP_USER, and SMTP_PASSWORD on Railway.",
            "user": user or None,
            "transport": "smtp",
        }

    try:
        if port == 465:
            context = ssl.create_default_context()
            with smtplib.SMTP_SSL(host, port, timeout=10, context=context) as server:
                server.login(user, password)
        else:
            with smtplib.SMTP(host, port, timeout=10) as server:
                server.ehlo()
                server.starttls(context=ssl.create_default_context())
                server.ehlo()
                server.login(user, password)
    except smtplib.SMTPAuthenticationError as exc:
        code = exc.smtp_code if hasattr(exc, "smtp_code") else None
        hint = (
            "Google rejected the login. The app password must be created while signed in as "
            f"{user} (with 2-Step Verification on). Delete the old app password and create a new one. "
            "If this is Google Workspace, your admin may need to allow app passwords."
        )
        return {"ok": False, "error": hint, "user": user, "smtp_code": code, "transport": "smtp"}
    except OSError as exc:
        hint = _smtp_network_error_hint(exc) or str(exc)
        return {"ok": False, "error": hint, "user": user, "transport": "smtp"}
    except Exception as exc:
        hint = _smtp_network_error_hint(exc) or str(exc)
        return {"ok": False, "error": hint, "user": user, "transport": "smtp"}

    return {"ok": True, "user": user, "host": host, "port": port, "transport": "smtp"}


def verify_email_connection() -> dict[str, Any]:
    transport = email_transport()
    if transport == "resend":
        return verify_resend_connection()
    if transport == "smtp":
        return verify_smtp_connection()
    return {
        "ok": False,
        "error": "Email not configured. Set RESEND_API_KEY or SMTP_* variables on Railway.",
        "transport": None,
    }


def _send_via_resend(to_email: str, subject: str, text: str, html: str) -> None:
    api_key = _resend_api_key()
    if not api_key:
        raise RuntimeError("RESEND_API_KEY is not set.")

    with httpx.Client(timeout=30.0) as client:
        response = client.post(
            "https://api.resend.com/emails",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "from": _from_header(),
                "to": [to_email],
                "subject": subject,
                "html": html,
                "text": text,
            },
        )

    if response.status_code >= 400:
        try:
            detail = response.json().get("message", response.text)
        except Exception:
            detail = response.text
        raise RuntimeError(f"Resend send failed: {detail}")


def _send_via_smtp(to_email: str, subject: str, text: str, html: str) -> None:
    settings = _smtp_settings()
    host = str(settings["host"])
    user = str(settings["user"])
    password = str(settings["password"])
    from_raw = str(settings["from_raw"])
    from_email = str(settings["from_email"])
    port = int(settings["port"])

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = from_raw if from_raw else formataddr(("Leads Wrestling", from_email))
    msg["To"] = to_email
    msg.attach(MIMEText(text, "plain", "utf-8"))
    msg.attach(MIMEText(html, "html", "utf-8"))

    try:
        if port == 465:
            context = ssl.create_default_context()
            with smtplib.SMTP_SSL(host, port, timeout=30, context=context) as server:
                server.login(user, password)
                server.sendmail(from_email, [to_email], msg.as_string())
        else:
            with smtplib.SMTP(host, port, timeout=30) as server:
                server.ehlo()
                server.starttls(context=ssl.create_default_context())
                server.ehlo()
                server.login(user, password)
                server.sendmail(from_email, [to_email], msg.as_string())
    except OSError as exc:
        hint = _smtp_network_error_hint(exc)
        if hint:
            raise RuntimeError(hint) from exc
        raise


def _lead_field(row: pd.Series | dict[str, Any], key: str) -> str:
    get = row.get if isinstance(row, dict) else row.get
    return _safe_str(get(key, ""))


def _lead_name(row: pd.Series | dict[str, Any]) -> str:
    first = _lead_field(row, "First Name")
    last = _lead_field(row, "Last Name")
    name = f"{first} {last}".strip()
    return name or _lead_field(row, "Email") or "Unknown lead"


def _esc(value: object) -> str:
    return html_module.escape(_safe_str(value))


def _nl2br(value: object) -> str:
    return _esc(value).replace("\n", "<br>")


_EMAIL_ACCENT = "#2563eb"


def _html_row(label: str, value: object) -> str:
    raw = _safe_str(value)
    if not raw:
        val = "—"
    elif "\n" in raw:
        val = _nl2br(value)
    else:
        val = _esc(value)
    return f"""
      <tr>
        <td style="padding:10px 12px;border-bottom:1px solid #e2e8f0;color:#64748b;font-size:13px;width:42%;vertical-align:top;">{label}</td>
        <td style="padding:10px 12px;border-bottom:1px solid #e2e8f0;color:#0f172a;font-size:14px;vertical-align:top;">{val}</td>
      </tr>"""


def brief_lines_for_row(row: pd.Series | dict[str, Any]) -> list[str]:
    """Pre-call brief lines from enrichment columns (empty if the lead was not enriched)."""
    get = row.get if isinstance(row, dict) else row.get
    summary = _safe_str(get("Enrichment Summary", ""))
    if not summary:
        return []
    lines = [ln.strip() for ln in summary.split("\n") if ln.strip()]
    facts: list[str] = []
    if _safe_str(get("TW Record", "")):
        rec = f"Record: {_safe_str(get('TW Record', ''))}"
        if _safe_str(get("TW Weight Class", "")):
            rec += f" @ {_safe_str(get('TW Weight Class', ''))}"
        if _safe_str(get("TW Team", "")):
            rec += f" ({_safe_str(get('TW Team', ''))})"
        facts.append(rec)
    if _safe_str(get("Club Affiliation", "")):
        facts.append(f"Club: {_safe_str(get('Club Affiliation', ''))}")
    confidence = _safe_str(get("Enrichment Confidence", ""))
    if confidence and confidence != "Error":
        facts.append(f"Confidence: {confidence}")
    return lines + facts


def _brief_links(row: pd.Series | dict[str, Any]) -> list[tuple[str, str]]:
    """(label, url) links to show under the brief — Flo profile first, then results source."""
    get = row.get if isinstance(row, dict) else row.get
    links: list[tuple[str, str]] = []
    profile = _safe_str(get("Flo Profile", ""))
    if profile:
        links.append(("FloWrestling profile", profile))
    source = _safe_str(get("TW Source URL", ""))
    if source and source != profile:
        links.append(("Results source", source))
    return links


def brief_note_html(row: pd.Series | dict[str, Any]) -> str:
    """Compact HTML for a HubSpot Note on the contact timeline."""
    lines = brief_lines_for_row(row)
    items = "".join(f"<li>{_esc(ln)}</li>" for ln in lines)
    links_html = "".join(f'<p><a href="{_esc(url)}">{_esc(label)}</a></p>' for label, url in _brief_links(row))
    return f"<p><strong>Pre-call brief (LeadsWrestling)</strong></p><ul>{items}</ul>{links_html}"


def _build_brief_html(row: pd.Series | dict[str, Any]) -> str:
    lines = brief_lines_for_row(row)
    if not lines:
        return ""
    get = row.get if isinstance(row, dict) else row.get
    is_error = _safe_str(get("Enrichment Confidence", "")) == "Error"
    heading = "Pre-call brief — unavailable" if is_error else "Pre-call brief"
    bg, border, head_color = ("#fef2f2", "#fecaca", "#991b1b") if is_error else ("#fffbeb", "#fcd34d", "#92400e")
    items = "".join(f'<li style="margin:0 0 6px;">{_esc(ln)}</li>' for ln in lines)
    source_html = "".join(
        f'<p style="margin:8px 0 0;font-size:12px;"><a href="{_esc(url)}" style="color:{_EMAIL_ACCENT};">{_esc(label)} →</a></p>'
        for label, url in _brief_links(row)
    )
    return f"""
            <table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="margin-bottom:20px;background:{bg};border:1px solid {border};border-radius:10px;">
              <tr>
                <td style="padding:14px 18px;">
                  <p style="margin:0 0 8px;font-size:12px;letter-spacing:0.06em;text-transform:uppercase;color:{head_color};font-weight:700;">{heading}</p>
                  <ul style="margin:0;padding-left:18px;font-size:14px;color:#1c1917;line-height:1.5;">{items}</ul>
                  {source_html}
                </td>
              </tr>
            </table>"""


def _build_assignment_html(
    *,
    rep_first: str,
    name: str,
    row: pd.Series | dict[str, Any],
    form_config: dict[str, Any] | None = None,
) -> str:
    form_rows = "".join(
        _html_row(label, val)
        for label, val in form_entries_for_row(row, form_config=form_config)
    )
    brief_html = _build_brief_html(row)

    return f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"></head>
<body style="margin:0;padding:0;background:#f1f5f9;font-family:Segoe UI,Helvetica,Arial,sans-serif;">
  <table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="background:#f1f5f9;padding:24px 12px;">
    <tr><td align="center">
      <table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="max-width:600px;background:#ffffff;border-radius:12px;overflow:hidden;border:1px solid #e2e8f0;">
        <tr>
          <td style="background:{_EMAIL_ACCENT};padding:20px 24px;">
            <p style="margin:0 0 6px;font-size:12px;letter-spacing:0.06em;text-transform:uppercase;color:rgba(255,255,255,0.85);">Leads Wrestling</p>
            <h1 style="margin:0;font-size:22px;line-height:1.3;color:#ffffff;">New lead assigned</h1>
          </td>
        </tr>
        <tr>
          <td style="padding:24px;">
            <p style="margin:0 0 16px;font-size:15px;color:#334155;">Hi {_esc(rep_first)},</p>
            <p style="margin:0 0 20px;font-size:15px;color:#334155;line-height:1.6;">
              A new lead has been assigned to you. Here is what they submitted on your form:
            </p>
            <table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="margin-bottom:20px;background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px;">
              <tr>
                <td style="padding:16px 18px;">
                  <p style="margin:0;font-size:20px;font-weight:700;color:#0f172a;">{_esc(name)}</p>
                </td>
              </tr>
            </table>{brief_html}
            <table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="border:1px solid #e2e8f0;border-radius:10px;overflow:hidden;margin-bottom:20px;">
              {form_rows}
            </table>
          </td>
        </tr>
        <tr>
          <td style="padding:16px 24px;background:#f8fafc;border-top:1px solid #e2e8f0;font-size:12px;color:#94a3b8;text-align:center;">
            Sent automatically by LeadsWrestling
          </td>
        </tr>
      </table>
    </td></tr>
  </table>
</body>
</html>"""


def build_assignment_email(
    row: pd.Series | dict[str, Any],
    rep: dict[str, Any],
    assignment: dict[str, Any],
    *,
    form_config: dict[str, Any] | None = None,
) -> tuple[str, str, str]:
    """Return subject, plain text body, html body."""
    from .lead_form_fields import resolve_form_config_for_row

    resolved_form = resolve_form_config_for_row(row, form_config)
    name = lead_display_name(row)
    form_label = _safe_str((resolved_form or {}).get("label"))
    if form_label:
        subject = f"New lead ({form_label}): {name}"
    else:
        subject = f"New wrestling lead: {name}"

    rep_first = _safe_str(rep.get("name", "there")).split()[0] or "there"

    lines = [f"Hi {rep_first},", "", "NEW LEAD ASSIGNED", "", name, ""]
    if form_label:
        lines.append(f"Form: {form_label}")
        lines.append("")
    brief = brief_lines_for_row(row)
    if brief:
        lines.append("PRE-CALL BRIEF")
        lines.extend(f"- {ln}" for ln in brief)
        lines.extend(f"{label}: {url}" for label, url in _brief_links(row))
        lines.append("")
    for label, val in form_entries_for_row(row, form_config=resolved_form):
        lines.append(f"{label}: {val}")
    lines.extend(["", "— LeadsWrestling"])
    text = "\n".join(lines)

    html = _build_assignment_html(
        rep_first=rep_first,
        name=name,
        row=row,
        form_config=resolved_form,
    )

    return subject, text, html


def send_lead_assignment_email(
    row: pd.Series | dict[str, Any],
    rep: dict[str, Any],
    assignment: dict[str, Any],
    *,
    form_config: dict[str, Any] | None = None,
) -> bool:
    """Send assignment email via Resend (HTTPS) or SMTP."""
    rep_email = _safe_str(rep.get("email"))
    transport = email_transport()

    if not transport:
        raise RuntimeError(
            "Email not configured. Set RESEND_API_KEY or SMTP_* variables on Railway."
        )

    to_email, sandbox_banner = _resolve_delivery(rep_email)
    if not to_email:
        raise RuntimeError("Rep must have an email, or set RESEND_SANDBOX_TO for sandbox mode.")

    subject, text, html = build_assignment_email(row, rep, assignment, form_config=form_config)
    if sandbox_banner:
        subject = f"[Sandbox · for {rep.get('name', rep_email)}] {subject}"
        text = f"{sandbox_banner}\n\n{text}"
        html = (
            f'<p style="background:#fff3cd;padding:10px;border-radius:6px;">{sandbox_banner}</p>{html}'
        )

    if transport == "resend":
        _send_via_resend(to_email, subject, text, html)
    else:
        _send_via_smtp(to_email, subject, text, html)

    return True
