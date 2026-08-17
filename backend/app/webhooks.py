"""Wufoo webhook handlers for live lead scoring."""

from __future__ import annotations

import logging
import os
from typing import Any

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request

from .integrations.wufoo import wufoo_payload_to_lead_row
from .store import is_synthetic_test_lead, store
from .webhook_diagnostics import log_webhook_event, recent_webhook_events

router = APIRouter(prefix="/webhooks", tags=["webhooks"])
logger = logging.getLogger(__name__)


def _verify_wufoo_secret(request: Request, payload: dict[str, Any] | None = None) -> None:
    secret = os.getenv("WUFOO_WEBHOOK_SECRET")
    if not secret:
        return

    provided = (
        request.headers.get("X-Wufoo-Webhook-Secret")
        or request.headers.get("Authorization")
        or request.query_params.get("secret")
    )
    if payload:
        provided = provided or payload.get("HandshakeKey") or payload.get("handshakeKey")

    if provided != secret:
        log_webhook_event(
            outcome="rejected",
            detail="invalid_secret",
            query_form=_safe_str(request.query_params.get("form")),
        )
        raise HTTPException(
            status_code=401,
            detail=(
                "Invalid Wufoo webhook secret. The Handshake Key on this Wufoo form must "
                "exactly match WUFOO_WEBHOOK_SECRET on Railway (same value on every form)."
            ),
        )


def _safe_str(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


async def _parse_wufoo_body(request: Request) -> dict[str, Any]:
    content_type = (request.headers.get("content-type") or "").lower()

    if "application/json" in content_type:
        body = await request.json()
        return body if isinstance(body, dict) else {"Fields": body}

    # Wufoo default: application/x-www-form-urlencoded
    form = await request.form()
    return {key: form.get(key) for key in form.keys()}


async def _score_wufoo_lead(
    row: dict[str, Any],
    use_llm: bool,
    form_config: dict[str, Any] | None = None,
) -> None:
    """Score in background — Wufoo expects a 2xx response within a few seconds."""
    try:
        result = await store.append_lead(row, use_llm=use_llm, form_config=form_config)
        logger.info(
            "Wufoo lead scored email=%s form=%s tier=%s routed=%s",
            result.get("email"),
            (form_config or {}).get("id"),
            result.get("ai_tier"),
            (result.get("routing") or {}).get("assigned"),
        )
    except Exception:
        logger.exception(
            "Wufoo background scoring failed for email=%s form=%s",
            row.get("Email"),
            (form_config or {}).get("id"),
        )


def _enrichment_status() -> dict[str, Any]:
    from .clay_notify import clay_callback_secret, clay_configured, enrichment_n8n_url
    from .enrichment import ENRICHMENT_MODEL, claude_enrichment_configured, enrichment_provider

    return {
        "provider": enrichment_provider(),
        "claude_configured": claude_enrichment_configured(),
        "claude_model": ENRICHMENT_MODEL,
        "clay_webhook_configured": clay_configured(),
        "callback_path": "/webhooks/clay-enrichment",
        "callback_secret_configured": bool(clay_callback_secret()),
        "n8n_enrichment_webhook_configured": bool(enrichment_n8n_url()),
    }


@router.get("/enrichment/status")
def enrichment_status() -> dict[str, Any]:
    """Diagnostics for lead enrichment (does not expose secrets)."""
    return _enrichment_status()


@router.get("/clay/status")
def clay_status() -> dict[str, Any]:
    """Back-compat alias for /webhooks/enrichment/status."""
    return _enrichment_status()


def _verify_enrichment_secret(request: Request) -> None:
    from .clay_notify import clay_callback_secret

    secret = clay_callback_secret()
    if not secret:
        return
    provided = request.headers.get("X-Clay-Secret") or request.headers.get("Authorization")
    if provided != secret:
        raise HTTPException(status_code=401, detail="Invalid enrichment secret.")


@router.post("/enrichment/run")
def enrichment_run(request: Request, email: str, wait: bool = True) -> dict[str, Any]:
    """
    Manually enrich a lead already in the cache with Claude (for testing / re-runs).
    Secured by CLAY_CALLBACK_SECRET (header X-Clay-Secret). ?wait=false returns immediately.
    """
    from .enrichment import claude_enrichment_configured, enrich_lead_with_claude, start_claude_enrichment
    from .routing_config import load_routing_config

    _verify_enrichment_secret(request)
    if not claude_enrichment_configured():
        raise HTTPException(status_code=400, detail="Set ANTHROPIC_API_KEY (and ENRICHMENT_PROVIDER=claude) on the server.")

    idx = store.find_lead_index(email=email)
    if idx is None:
        raise HTTPException(status_code=404, detail=f"No lead in cache with email {email}.")
    row = store.get_row_at(idx)
    config = load_routing_config()
    rep = next(
        (r for r in config.get("reps", []) if _safe_str(r.get("email")) == _safe_str(row.get("Assigned Email"))),
        None,
    )
    if not wait:
        start_claude_enrichment(row, rep)
        return {"success": True, "email": email, "queued": True}
    try:
        outcome = enrich_lead_with_claude(row, rep)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Enrichment failed: {exc}") from exc
    return {"success": True, "email": email, **outcome}


@router.post("/clay-enrichment")
async def clay_enrichment_callback(request: Request) -> dict[str, Any]:
    """
    Receive enrichment results from Clay's HTTP API column, store them on the lead,
    and forward a pre-call brief to n8n (N8N_ENRICHMENT_WEBHOOK_URL) for the assigned rep.
    """
    from .enrichment import record_enrichment

    _verify_enrichment_secret(request)

    try:
        body = await request.json()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {exc}") from exc
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="Expected a JSON object.")

    email = _safe_str(body.get("email"))
    if not email:
        raise HTTPException(status_code=400, detail="Missing email.")

    outcome = record_enrichment(email, body, rep_id=_safe_str(body.get("rep_id")))
    if not outcome.get("stored"):
        log_webhook_event(outcome="rejected", detail="clay_enrichment_lead_not_found", email=email)
        raise HTTPException(status_code=404, detail=outcome.get("reason", "Lead not found."))

    log_webhook_event(outcome="accepted", detail="clay_enrichment", email=email)
    return {"success": True, "email": email, **{k: v for k, v in outcome.items() if k != "stored"}}


@router.get("/wufoo/status")
def wufoo_webhook_status() -> dict[str, Any]:
    """Diagnostics for Wufoo integration (does not expose secrets)."""
    from .integrations.wufoo import WOOFOO_MAP_PATH, load_wufoo_map
    from .integrations.wufoo_fields import (
        CACHE_PATH,
        load_cached_field_map,
        wufoo_api_configured,
    )

    field_map = load_wufoo_map()
    cached = load_cached_field_map()
    secret = os.getenv("WUFOO_WEBHOOK_SECRET")
    return {
        "webhook_path": "/webhooks/wufoo",
        "secret_configured": bool(secret),
        "field_map_loaded": bool(field_map),
        "field_map_path": str(WOOFOO_MAP_PATH),
        "mapped_field_count": len(field_map),
        "api_field_cache_count": len(cached),
        "api_field_cache_path": str(CACHE_PATH),
        "wufoo_api_configured": wufoo_api_configured(),
        "cache_loaded": store.loaded,
        "cache_row_count": int(store._meta.get("row_count", 0)) if store.loaded else 0,
        "last_scored_at": store._meta.get("last_append") or store._meta.get("scored_at") if store.loaded else None,
        "recent_webhook_events": recent_webhook_events(8),
    }


@router.post("/wufoo/sync-fields")
async def wufoo_sync_fields(request: Request) -> dict[str, Any]:
    """Refresh FieldN → column map from Wufoo API (requires WUFOO_API_KEY on Railway)."""
    from .integrations.wufoo_fields import sync_wufoo_field_cache, wufoo_api_configured

    if not wufoo_api_configured():
        raise HTTPException(
            status_code=400,
            detail="Set WUFOO_API_KEY, WUFOO_SUBDOMAIN, and WUFOO_FORM on the server.",
        )
    payload: dict[str, Any] = {}
    if request.headers.get("content-type", "").startswith("application/json"):
        try:
            body = await request.json()
            if isinstance(body, dict):
                payload = body
        except Exception:
            payload = {}
    _verify_wufoo_secret(request, payload if payload else None)
    field_map = sync_wufoo_field_cache()
    return {"ok": True, "mapped_field_count": len(field_map)}


@router.get("/wufoo/forms")
def wufoo_forms_list() -> dict[str, Any]:
    """Per-form routing config and webhook URL hints."""
    from .wufoo_forms import get_form, list_forms_public, load_forms_config, webhook_url_hint

    base = os.getenv("PUBLIC_API_URL", "").strip() or "https://wrestlingleads-production.up.railway.app"
    forms = list_forms_public()
    for summary in forms:
        full = get_form(str(summary.get("id", ""))) or summary
        summary["webhook_url_example"] = webhook_url_hint(base, full, secret="YOUR_HANDSHAKE_KEY")
    return {
        "forms": forms,
        "default_form_id": load_forms_config().get("default_form_id", "form-1"),
        "policies": {
            "ai": "Score with AI + Team distribution rules + n8n",
            "fixed_reps": "Always assign to fixed_rep_ids (round robin)",
            "off": "Store/score only — no route or n8n",
        },
    }


@router.post("/wufoo")
async def wufoo_webhook(
    request: Request,
    background_tasks: BackgroundTasks,
    use_llm: bool = True,
    form: str | None = None,
) -> dict[str, Any]:
    """
    Receive a Wufoo form submission, score it, and append to dashboard cache.

    Configure in Wufoo: Form → More → Integrations → WebHook (paid plans only)
    URL: https://your-api/webhooks/wufoo
    Handshake Key: match WUFOO_WEBHOOK_SECRET in Railway/.env
    """
    try:
        payload = await _parse_wufoo_body(request)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid webhook payload: {exc}") from exc

    _verify_wufoo_secret(request, payload)

    from .wufoo_forms import resolve_form

    form_config = resolve_form(query_form=form, payload=payload)
    row = wufoo_payload_to_lead_row(payload, form_config=form_config or None)
    if is_synthetic_test_lead(row):
        log_webhook_event(
            outcome="rejected",
            detail="synthetic_test_lead",
            entry_id=_safe_str(payload.get("EntryId") or payload.get("EntryID")),
            email=_safe_str(row.get("Email")),
            form_id=_safe_str((form_config or {}).get("id")),
            query_form=_safe_str(form),
        )
        raise HTTPException(status_code=400, detail="Synthetic test submissions are not stored.")

    if not row.get("Email") and not row.get("Message"):
        from .integrations.wufoo import WOOFOO_MAP_PATH, load_wufoo_map

        if not load_wufoo_map(form_config) and not form_config:
            raise HTTPException(
                status_code=500,
                detail=f"Wufoo field map missing on server ({WOOFOO_MAP_PATH}). Redeploy latest Docker image.",
            )
        log_webhook_event(
            outcome="rejected",
            detail="missing_email_and_message",
            entry_id=_safe_str(payload.get("EntryId") or payload.get("EntryID")),
            form_id=_safe_str((form_config or {}).get("id")),
            query_form=_safe_str(form),
        )
        raise HTTPException(
            status_code=400,
            detail="Webhook missing mappable lead fields. Check wufoo_forms.json for this form.",
        )

    entry_id = payload.get("EntryId") or payload.get("EntryID") or row.get("Record ID")
    log_webhook_event(
        outcome="accepted",
        entry_id=_safe_str(entry_id),
        email=_safe_str(row.get("Email")),
        form_id=_safe_str((form_config or {}).get("id")),
        query_form=_safe_str(form),
    )
    logger.info(
        "Wufoo webhook accepted entry=%s email=%s form=%s",
        entry_id,
        row.get("Email"),
        (form_config or {}).get("id"),
    )

    routing = (form_config or {}).get("routing") or {}
    score_with_ai = routing.get("score_with_ai", True)

    # Reply immediately — Wufoo times out if DeepSeek scoring blocks the HTTP response.
    background_tasks.add_task(_score_wufoo_lead, row, use_llm and score_with_ai, form_config)

    return {
        "success": True,
        "status": "accepted",
        "entry_id": entry_id,
        "email": row.get("Email"),
        "form_id": (form_config or {}).get("id"),
        "message": "Lead queued for scoring — check dashboard in ~30 seconds",
    }
