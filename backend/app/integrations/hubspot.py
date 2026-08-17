"""Sync routed leads to HubSpot CRM (create or update contact by email / Record ID)."""

from __future__ import annotations

import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import pandas as pd

from ..features import _safe_str

BASE_DIR = Path(__file__).resolve().parents[3]
HUBSPOT_MAP_PATH = BASE_DIR / "config" / "hubspot_field_map.json"
HUBSPOT_API = "https://api.hubapi.com/crm/v3/objects/contacts"
HUBSPOT_OWNERS_API = "https://api.hubapi.com/crm/v3/owners"
_OWNERS_CACHE_TTL_SEC = 300
_owners_by_email_cache: dict[str, str] = {}
_owners_cache_loaded_at: float = 0.0


def hubspot_configured() -> bool:
    return bool(os.getenv("HUBSPOT_ACCESS_TOKEN", "").strip())


def load_hubspot_map() -> dict[str, str]:
    data = json.loads(HUBSPOT_MAP_PATH.read_text(encoding="utf-8"))
    return dict(data.get("source_to_hubspot_field_map", {}))


def lead_row_to_hubspot_properties(row: pd.Series | dict[str, Any]) -> dict[str, str]:
    """Map a scored lead row to HubSpot contact properties (string values)."""
    field_map = load_hubspot_map()
    properties: dict[str, str] = {}

    if isinstance(row, pd.Series):
        row = row.to_dict()

    for source_col, hubspot_prop in field_map.items():
        value = row.get(source_col)
        if value is None or (isinstance(value, float) and pd.isna(value)):
            continue
        text = str(value).strip()
        if text:
            properties[hubspot_prop] = text

    return properties


def _fetch_owners_by_email(token: str) -> dict[str, str]:
    """Map HubSpot owner user email → owner id (paginated)."""
    by_email: dict[str, str] = {}
    after: str | None = None
    with httpx.Client(timeout=20.0) as client:
        while True:
            params: dict[str, str | int] = {"limit": 100}
            if after:
                params["after"] = after
            response = client.get(
                HUBSPOT_OWNERS_API,
                headers=_headers(token),
                params=params,
            )
            if response.status_code >= 400:
                raise RuntimeError(_api_error(response))
            body = response.json()
            for row in body.get("results", []):
                owner_id = _safe_str(row.get("id"))
                email = _safe_str(row.get("email")).lower()
                if owner_id and email:
                    by_email[email] = owner_id
            after = (body.get("paging") or {}).get("next", {}).get("after")
            if not after:
                break
    return by_email


def _owners_by_email() -> dict[str, str]:
    global _owners_by_email_cache, _owners_cache_loaded_at
    token = os.getenv("HUBSPOT_ACCESS_TOKEN", "").strip()
    if not token:
        return {}
    now = time.time()
    if _owners_by_email_cache and now - _owners_cache_loaded_at < _OWNERS_CACHE_TTL_SEC:
        return _owners_by_email_cache
    _owners_by_email_cache = _fetch_owners_by_email(token)
    _owners_cache_loaded_at = now
    return _owners_by_email_cache


def hubspot_owner_resolve_note(rep: dict[str, Any]) -> str:
    """Human-readable reason owner id is present or missing (for n8n webhook debug)."""
    manual = _normalize_owner_id(rep.get("hubspot_owner_id"))
    if manual:
        return f"Using HubSpot owner ID {manual} from Team settings."
    rep_email = _safe_str(rep.get("email")).lower()
    if not rep_email:
        return "Rep has no email on Team — add email or HubSpot owner ID."
    if not hubspot_configured():
        return (
            "HubSpot is handled in n8n, not on Railway. Paste this rep's HubSpot owner ID on "
            "Team → Save (HubSpot → Settings → Users → open user → numeric ID in the URL)."
        )
    try:
        owners = _owners_by_email()
        if rep_email in owners:
            return f"Matched HubSpot user {rep_email} → owner {owners[rep_email]}."
        emails = ", ".join(sorted(owners.keys())[:8])
        suffix = "…" if len(owners) > 8 else ""
        return (
            f"No HubSpot user for {rep_email}. Token sees: {emails}{suffix}. "
            "Fix Team email or paste owner ID on Team."
        )
    except Exception as exc:
        return f"HubSpot owners API failed: {exc}"


def _normalize_owner_id(value: object) -> str:
    """Digits-only HubSpot owner id (handles str/int from Team JSON)."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return ""
    if isinstance(value, int):
        return str(value) if value > 0 else ""
    if isinstance(value, float) and value == int(value):
        return str(int(value))
    raw = _safe_str(value)
    if raw.isdigit():
        return raw
    digits = "".join(c for c in raw if c.isdigit())
    return digits


def hubspot_owner_id_for_rep(rep: dict[str, Any]) -> str:
    """
    Contact owner on sync: explicit hubspot_owner_id on the rep, else match rep email
    to a HubSpot owner user (requires HUBSPOT_ACCESS_TOKEN on Railway).
    """
    manual = _normalize_owner_id(rep.get("hubspot_owner_id"))
    if manual:
        return manual
    rep_email = _safe_str(rep.get("email")).lower()
    if not rep_email or not hubspot_configured():
        return ""
    try:
        return _owners_by_email().get(rep_email, "")
    except Exception:
        return ""


def list_hubspot_owners() -> list[dict[str, str]]:
    """All HubSpot owners for Team UI / troubleshooting."""
    if not hubspot_configured():
        return []
    token = os.getenv("HUBSPOT_ACCESS_TOKEN", "").strip()
    owners: list[dict[str, str]] = []
    after: str | None = None
    with httpx.Client(timeout=20.0) as client:
        while True:
            params: dict[str, str | int] = {"limit": 100}
            if after:
                params["after"] = after
            response = client.get(
                HUBSPOT_OWNERS_API,
                headers=_headers(token),
                params=params,
            )
            if response.status_code >= 400:
                raise RuntimeError(_api_error(response))
            body = response.json()
            for row in body.get("results", []):
                oid = _safe_str(row.get("id"))
                email = _safe_str(row.get("email"))
                first = _safe_str(row.get("firstName"))
                last = _safe_str(row.get("lastName"))
                name = f"{first} {last}".strip() or email
                if oid:
                    owners.append({"id": oid, "email": email, "name": name})
            after = (body.get("paging") or {}).get("next", {}).get("after")
            if not after:
                break
    return owners


def _assignment_properties(
    rep: dict[str, Any],
    assignment: dict[str, Any],
) -> dict[str, str]:
    props: dict[str, str] = {
        "lw_assigned_rep": _safe_str(rep.get("name")),
        "lw_route_reason": _safe_str(assignment.get("route_reason")),
        "lw_assigned_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    owner_id = hubspot_owner_id_for_rep(rep)
    if owner_id:
        props["hubspot_owner_id"] = owner_id
    return {k: v for k, v in props.items() if v}


def build_route_sync_properties(
    row: pd.Series | dict[str, Any],
    rep: dict[str, Any],
    assignment: dict[str, Any],
) -> dict[str, str]:
    props = lead_row_to_hubspot_properties(row)
    props.update(_assignment_properties(rep, assignment))
    email = props.get("email") or _lead_email(row)
    if email:
        props["email"] = email
    return props


def _lead_email(row: pd.Series | dict[str, Any]) -> str:
    get = row.get if isinstance(row, dict) else row.get
    return _safe_str(get("Email", ""))


def _hubspot_record_id(row: pd.Series | dict[str, Any]) -> str:
    get = row.get if isinstance(row, dict) else row.get
    raw = _safe_str(get("Record ID", ""))
    return raw if raw.isdigit() else ""


def _headers(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}


def _api_error(response: httpx.Response) -> str:
    try:
        body = response.json()
        message = body.get("message") or body.get("detail")
        if message:
            return str(message)
    except Exception:
        pass
    return response.text[:300] or f"HTTP {response.status_code}"


def _patch_contact(
    client: httpx.Client,
    *,
    token: str,
    contact_id: str,
    properties: dict[str, str],
) -> dict[str, Any]:
    response = client.patch(
        f"{HUBSPOT_API}/{contact_id}",
        headers=_headers(token),
        json={"properties": properties},
    )
    if response.status_code == 404:
        raise LookupError(contact_id)
    if response.status_code >= 400:
        raise RuntimeError(_api_error(response))
    return {"action": "updated", "id": contact_id}


def _search_contact_by_email(
    client: httpx.Client,
    *,
    token: str,
    email: str,
) -> str | None:
    response = client.post(
        f"{HUBSPOT_API}/search",
        headers=_headers(token),
        json={
            "filterGroups": [
                {
                    "filters": [
                        {"propertyName": "email", "operator": "EQ", "value": email}
                    ]
                }
            ],
            "limit": 1,
        },
    )
    if response.status_code >= 400:
        raise RuntimeError(_api_error(response))
    results = response.json().get("results", [])
    if not results:
        return None
    return str(results[0]["id"])


def create_or_update_contact(properties: dict[str, str]) -> dict[str, Any]:
    """
    Create or update a HubSpot contact.
    Uses Record ID when present, otherwise email search, otherwise create.
    """
    token = os.getenv("HUBSPOT_ACCESS_TOKEN", "").strip()
    if not token:
        raise RuntimeError("HUBSPOT_ACCESS_TOKEN is not configured.")

    email = _safe_str(properties.get("email"))
    record_id = _safe_str(properties.get("_record_id", ""))
    if record_id:
        del properties["_record_id"]

    if not email and not record_id:
        raise RuntimeError("Lead must have an email (or HubSpot Record ID) to sync to HubSpot.")

    with httpx.Client(timeout=30.0) as client:
        if record_id:
            try:
                return _patch_contact(
                    client, token=token, contact_id=record_id, properties=properties
                )
            except LookupError:
                pass

        if email:
            existing_id = _search_contact_by_email(client, token=token, email=email)
            if existing_id:
                return _patch_contact(
                    client, token=token, contact_id=existing_id, properties=properties
                )

        if not email:
            raise RuntimeError(
                "HubSpot contact not found for Record ID and no email to create a new contact."
            )

        response = client.post(
            HUBSPOT_API,
            headers=_headers(token),
            json={"properties": properties},
        )
        if response.status_code >= 400:
            raise RuntimeError(_api_error(response))
        body = response.json()
        return {"action": "created", "id": str(body.get("id", ""))}


HUBSPOT_NOTES_API = "https://api.hubapi.com/crm/v3/objects/notes"
_NOTE_TO_CONTACT_ASSOCIATION_TYPE_ID = 202


def _ensure_contact_id(client: httpx.Client, *, token: str, email: str, properties: dict[str, str]) -> str:
    """Find the contact by email, or create a minimal one. Handles the create race (409 → existing id)."""
    existing = _search_contact_by_email(client, token=token, email=email)
    if existing:
        return existing
    response = client.post(HUBSPOT_API, headers=_headers(token), json={"properties": {"email": email, **properties}})
    if response.status_code == 409:
        # "Contact already exists. Existing ID: 12345"
        import re

        match = re.search(r"Existing ID:\s*(\d+)", _api_error(response))
        if match:
            return match.group(1)
    if response.status_code >= 400:
        raise RuntimeError(_api_error(response))
    return str(response.json().get("id", ""))


def add_enrichment_note(
    email: str,
    note_html: str,
    *,
    first_name: str = "",
    last_name: str = "",
) -> dict[str, Any]:
    """Attach the pre-call brief to the contact's timeline as a Note (no custom properties needed)."""
    token = os.getenv("HUBSPOT_ACCESS_TOKEN", "").strip()
    if not token:
        raise RuntimeError("HUBSPOT_ACCESS_TOKEN is not configured.")
    if not email:
        raise RuntimeError("Lead must have an email to attach a HubSpot note.")

    props: dict[str, str] = {}
    if first_name:
        props["firstname"] = first_name
    if last_name:
        props["lastname"] = last_name

    with httpx.Client(timeout=30.0) as client:
        contact_id = _ensure_contact_id(client, token=token, email=email, properties=props)
        response = client.post(
            HUBSPOT_NOTES_API,
            headers=_headers(token),
            json={
                "properties": {
                    "hs_timestamp": str(int(time.time() * 1000)),
                    "hs_note_body": note_html,
                },
                "associations": [
                    {
                        "to": {"id": contact_id},
                        "types": [
                            {
                                "associationCategory": "HUBSPOT_DEFINED",
                                "associationTypeId": _NOTE_TO_CONTACT_ASSOCIATION_TYPE_ID,
                            }
                        ],
                    }
                ],
            },
        )
        if response.status_code >= 400:
            raise RuntimeError(_api_error(response))
        return {"contact_id": contact_id, "note_id": str(response.json().get("id", ""))}


def sync_contact_on_route(
    row: pd.Series | dict[str, Any],
    rep: dict[str, Any],
    assignment: dict[str, Any],
) -> dict[str, Any]:
    """Upsert HubSpot contact when a lead is assigned."""
    properties = build_route_sync_properties(row, rep, assignment)
    record_id = _hubspot_record_id(row)
    if record_id:
        properties["_record_id"] = record_id
    result = create_or_update_contact(properties)
    owner_id = hubspot_owner_id_for_rep(rep)
    if owner_id:
        result["hubspot_owner_id"] = owner_id
    elif _safe_str(rep.get("email")):
        result["hubspot_owner_note"] = (
            f"No HubSpot owner user found for {_safe_str(rep.get('email'))} — "
            "add hubspot_owner_id on the rep or use the same email in HubSpot Users."
        )
    return result


def verify_hubspot_connection() -> dict[str, Any]:
    if not hubspot_configured():
        return {
            "ok": False,
            "error": "HUBSPOT_ACCESS_TOKEN is not set on the server.",
            "transport": "hubspot",
        }

    token = os.getenv("HUBSPOT_ACCESS_TOKEN", "").strip()
    try:
        with httpx.Client(timeout=20.0) as client:
            response = client.get(
                "https://api.hubapi.com/crm/v3/owners",
                headers=_headers(token),
                params={"limit": 1},
            )
        if response.status_code >= 400:
            return {"ok": False, "error": _api_error(response), "transport": "hubspot"}
    except Exception as exc:
        return {"ok": False, "error": str(exc), "transport": "hubspot"}

    owner_count = 0
    try:
        owner_count = len(_owners_by_email())
    except Exception:
        pass

    return {
        "ok": True,
        "transport": "hubspot",
        "note": (
            f"Token OK — {owner_count} HubSpot owner(s) loaded. "
            "On route, contact owner is set from each rep's hubspot_owner_id or rep email match."
        ),
    }
