"""Assign scored leads to sales reps and notify by email."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd

from .features import (
    URGENT_DEADLINE_KEYWORDS,
    _safe_str,
    has_coaching_signals,
    is_high_investment_level,
    is_low_investment_level,
    is_near_term_deadline,
    is_parent_icp_buyer,
    is_soft_mental_edge_inquiry,
    is_sparse_subscriber,
    is_struggling_mentally,
)
from .routing_config import get_rep_by_id, load_routing_config, reps_for_bucket
from .scoring_config import get_tier_thresholds
from .routing_log import append_routing_entry, consecutive_routes_to_rep, count_rep_this_week, was_lead_routed
from .routing_notify import send_lead_assignment_email, email_configured
from .integrations.hubspot import hubspot_configured, sync_contact_on_route
from .n8n_notify import n8n_configured, send_n8n_assignment_notification
from .enrichment import enrich_before_notify, enrichment_enabled

WEST_COAST_STATE_CODES = frozenset({"CA", "OR", "WA", "NV", "AZ", "HI", "AK"})
WEST_COAST_REP_ID = "eric"
WEST_COAST_NAME_HINTS = (
    "california",
    "oregon",
    "washington",
    "nevada",
    "arizona",
    "hawaii",
    "alaska",
)


def _score(row: pd.Series | dict[str, Any]) -> float:
    get = row.get if isinstance(row, dict) else row.get
    try:
        return float(get("AI Score", 0) or 0)
    except (TypeError, ValueError):
        return 0.0


def _tier(row: pd.Series | dict[str, Any]) -> str:
    get = row.get if isinstance(row, dict) else row.get
    return _safe_str(get("AI Tier", ""))


def _assigned_reason(rep: dict[str, Any], note: str = "") -> str:
    name = _safe_str(rep.get("name")) or "rep"
    if note:
        return f"Assigned to {name} ({note})"
    return f"Assigned to {name}"


def _is_ready_now(row: pd.Series | dict[str, Any]) -> bool:
    get = row.get if isinstance(row, dict) else row.get
    rel = _safe_str(get("Relationship Status", "")).lower()
    if "ready to start" in rel:
        return True
    deadline = _safe_str(get("Deadline for Goal", "")).lower()
    return any(k in deadline for k in URGENT_DEADLINE_KEYWORDS)


def _normalize_state(value: object) -> str:
    text = _safe_str(value).upper()
    if not text:
        return ""
    if len(text) == 2 and text.isalpha():
        return text
    for token in text.replace(",", " ").split():
        if len(token) == 2 and token.isalpha():
            return token.upper()
    return text[:2] if len(text) >= 2 else text


def _normalize_city_state_country(value: object) -> str:
    """Extract state/region from a City, State, Country string."""
    text = _safe_str(value)
    if not text:
        return ""
    parts = [p.strip() for p in text.split(",")]
    if len(parts) >= 2:
        for part in parts[1:]:
            part_upper = part.strip().upper()
            if len(part_upper) == 2 and part_upper.isalpha():
                return part_upper
    return ""


def is_west_coast(row: pd.Series | dict[str, Any]) -> bool:
    get = row.get if isinstance(row, dict) else row.get
    state = _normalize_state(get("State/Region", ""))
    if state in WEST_COAST_STATE_CODES:
        return True
    region = _safe_str(get("State/Region", "")).lower()
    if any(hint in region for hint in WEST_COAST_NAME_HINTS):
        return True
    csc = _normalize_city_state_country(get("City/State/Country", ""))
    if csc in WEST_COAST_STATE_CODES:
        return True
    csc_full = _safe_str(get("City/State/Country", "")).lower()
    return any(hint in csc_full for hint in WEST_COAST_NAME_HINTS)


def _gene_urgency_signals(row: pd.Series | dict[str, Any]) -> bool:
    """Gene urgent queue: parent buyers, real urgency, or serious mental struggle — not budget self-signups."""
    if is_low_investment_level(row) and not is_near_term_deadline(row):
        return False
    if is_soft_mental_edge_inquiry(row) and not is_high_investment_level(row):
        return False
    if is_parent_icp_buyer(row):
        return True
    if is_near_term_deadline(row):
        return True
    if is_high_investment_level(row):
        return True
    if is_struggling_mentally(row):
        return True
    return False


def is_urgent_red_hot(row: pd.Series | dict[str, Any], config: dict[str, Any]) -> bool:
    """Gene bucket: Hot tier, high score, ready soon, and real urgency signals."""
    tier = _tier(row)
    score = _score(row)
    if tier != "Hot":
        return False
    hot_floor = float(get_tier_thresholds()["Hot"])
    min_score = float(config.get("urgent_min_score", hot_floor))
    if score < min_score:
        return False
    if not _is_ready_now(row):
        return False
    return _gene_urgency_signals(row)


def is_jake_tier(row: pd.Series | dict[str, Any], config: dict[str, Any]) -> bool:
    tier = _tier(row)
    score = _score(row)
    if tier == "Hot":
        return True
    min_warm = float(config.get("jake_min_warm_score", 70))
    return tier == "Warm" and score >= min_warm


def should_route_lead(row: pd.Series | dict[str, Any]) -> tuple[bool, str]:
    if is_sparse_subscriber(row):
        return False, "Sparse subscriber — no coaching outreach"
    if not has_coaching_signals(row):
        return False, "No coaching form data"
    lifecycle = _safe_str(row.get("Lifecycle Stage", "") if isinstance(row, dict) else row.get("Lifecycle Stage", ""))
    if lifecycle == "Customer":
        return False, "Already a customer"
    tier = _tier(row)
    if tier == "Unqualified":
        return False, "Unqualified tier"
    return True, ""


def should_route_lead_for_form(
    row: pd.Series | dict[str, Any],
    form_config: dict[str, Any] | None,
) -> tuple[bool, str]:
    """Per-form gate before assignment (fixed-rep forms can relax coaching-signal checks)."""
    routing = (form_config or {}).get("routing") or {}
    policy = _safe_str(routing.get("policy", "ai")).lower()
    if policy == "off":
        return False, "Form configured for intake only (no routing)"

    get = row.get if isinstance(row, dict) else row.get
    lifecycle = _safe_str(get("Lifecycle Stage", ""))
    if lifecycle == "Customer":
        return False, "Already a customer"

    if policy == "fixed_reps" and not routing.get("require_coaching_signals", False):
        if not _safe_str(get("Email", "")):
            return False, "No email on submission"
        return True, ""

    return should_route_lead(row)


def _pick_fixed_rep(
    config: dict[str, Any],
    form_config: dict[str, Any],
    row: pd.Series | dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    routing = form_config.get("routing") or {}
    region_map = routing.get("region_routing")

    if region_map and row is not None:
        region_id = "west" if is_west_coast(row) else "east"
        rep_ids = list(region_map.get(region_id) or region_map.get("east") or [])
    else:
        rep_ids = list(routing.get("fixed_rep_ids") or routing.get("rep_ids") or [])

    if not rep_ids:
        return None

    candidates: list[dict[str, Any]] = []
    for rep_id in rep_ids:
        rep = get_rep_by_id(config, _safe_str(rep_id))
        if rep and _rep_under_cap(rep):
            candidates.append(rep)
    if not candidates:
        for rep_id in rep_ids:
            rep = get_rep_by_id(config, _safe_str(rep_id))
            if rep:
                candidates.append(rep)
    if not candidates:
        return None

    pick_mode = _safe_str(routing.get("fixed_rep_pick", "round_robin")).lower()
    if pick_mode == "first":
        return candidates[0]

    counts = {rid: count_rep_this_week(rid) for rid in rep_ids}
    return min(candidates, key=lambda rep: counts.get(_safe_str(rep.get("id")), 0))


def _assign_fixed_reps(
    row: pd.Series | dict[str, Any],
    config: dict[str, Any],
    form_config: dict[str, Any],
) -> dict[str, Any]:
    rep = _pick_fixed_rep(config, form_config, row=row)
    if not rep:
        return {
            "assigned": False,
            "skipped_reason": "Fixed-rep form has no matching reps on Team",
        }
    label = _safe_str(form_config.get("label")) or _safe_str(form_config.get("id"))
    region_map = (form_config.get("routing") or {}).get("region_routing")
    note = f"Form: {label}"
    if region_map:
        region_label = "West Coast" if is_west_coast(row) else "East Coast"
        note = f"Form: {label} ({region_label})"
    return {
        "assigned": True,
        "rep": rep,
        "route_bucket": "form_fixed",
        "route_reason": _assigned_reason(rep, note),
        "distribution_band": "form_fixed",
        "form_id": _safe_str(form_config.get("id")),
    }


def _form_routing_policy(form_config: dict[str, Any] | None) -> str:
    if not form_config:
        return "ai"
    return _safe_str((form_config.get("routing") or {}).get("policy", "ai")).lower()


def _form_send_to_n8n(form_config: dict[str, Any] | None) -> bool:
    if not form_config:
        return True
    return bool((form_config.get("routing") or {}).get("send_to_n8n", True))


def _form_auto_route(form_config: dict[str, Any] | None, config: dict[str, Any]) -> bool:
    if form_config:
        routing = form_config.get("routing") or {}
        if "auto_route" in routing:
            return bool(routing.get("auto_route"))
    return bool(config.get("auto_route_enabled", True))


def _rep_under_cap(rep: dict[str, Any]) -> bool:
    cap = rep.get("weekly_cap")
    if cap is None:
        return True
    try:
        cap_n = int(cap)
    except (TypeError, ValueError):
        return True
    return count_rep_this_week(_safe_str(rep.get("id"))) < cap_n


def _distribution_pcts(config: dict[str, Any]) -> tuple[float, float, float, float]:
    """Gene top %, Jake next %, general next %, automation bottom % (must sum ~100)."""
    gene = float(config.get("distribution_gene_pct", 10))
    jake = float(config.get("distribution_jake_pct", 20))
    general = float(config.get("distribution_general_pct", 50))
    automation = float(config.get("distribution_automation_pct", 20))
    total = gene + jake + general + automation
    if total <= 0:
        return 10.0, 20.0, 50.0, 20.0
    scale = 100.0 / total
    return gene * scale, jake * scale, general * scale, automation * scale


def _routable_scores_from_store() -> list[float]:
    from .store import store

    if not store.loaded:
        return []
    scores: list[float] = []
    try:
        row_count = len(store._df)  # type: ignore[arg-type]
    except Exception:
        return []
    for i in range(row_count):
        row = store.get_row_at(i)
        ok, _ = should_route_lead(row)
        if ok:
            scores.append(_score(row))
    return scores


def _percentile_band(score: float, config: dict[str, Any]) -> Literal["gene", "jake", "general", "automation"]:
    scores = _routable_scores_from_store()
    min_leads = int(config.get("min_leads_for_percentile", 15))
    if len(scores) < min_leads:
        return "general"

    gene_pct, jake_pct, general_pct, automation_pct = _distribution_pcts(config)
    # Cumulative cutoffs from the bottom (automation = lowest scores).
    p_auto = float(np.percentile(scores, automation_pct))
    p_general = float(np.percentile(scores, automation_pct + general_pct))
    p_jake = float(np.percentile(scores, automation_pct + general_pct + jake_pct))

    if score < p_auto:
        return "automation"
    if score < p_general:
        return "general"
    if score < p_jake:
        return "jake"
    return "gene"


def _first_rep_with_cap(
    reps: list[dict[str, Any]],
) -> dict[str, Any] | None:
    for rep in reps:
        if _rep_under_cap(rep):
            return rep
    return None


def _try_pick_jake(
    row: pd.Series | dict[str, Any],
    config: dict[str, Any],
) -> tuple[dict[str, Any] | None, str]:
    """Jake only when lead meets tier bar, weekly cap, and consecutive-route limit."""
    if not is_jake_tier(row, config):
        min_warm = float(config.get("jake_min_warm_score", 70))
        tier = _tier(row)
        if tier == "Warm":
            return None, f"Warm lead below Jake bar (≥{min_warm:.0f})"
        return None, "Jake gets Hot or strong Warm only"

    jake_reps = reps_for_bucket(config, "hot_warm")
    jake = _first_rep_with_cap(jake_reps)
    if not jake:
        return None, "Jake at weekly cap"

    max_consec = int(config.get("jake_max_consecutive", 2))
    if max_consec > 0:
        streak = consecutive_routes_to_rep(_safe_str(jake.get("id")))
        if streak >= max_consec:
            return None, f"Jake limit ({max_consec} in a row) — general pool"

    return jake, ""


def _pick_general_rep(row: pd.Series | dict[str, Any], config: dict[str, Any]) -> tuple[dict[str, Any], str]:
    general = reps_for_bucket(config, "general")
    if not general:
        raise ValueError("No general-pool reps configured.")

    west = is_west_coast(row)
    eric = next((r for r in general if _safe_str(r.get("id")) == WEST_COAST_REP_ID), None)
    beau = next((r for r in general if _safe_str(r.get("id")) != WEST_COAST_REP_ID), None)

    if west and eric and _rep_under_cap(eric):
        return eric, _assigned_reason(eric, "West Coast")

    candidates = [r for r in general if r]
    if len(candidates) == 1:
        return candidates[0], _assigned_reason(candidates[0])

    counts = {r["id"]: count_rep_this_week(_safe_str(r.get("id"))) for r in candidates}
    chosen = min(candidates, key=lambda r: counts.get(r["id"], 0))
    if west and eric and _rep_under_cap(eric) and chosen.get("id") != eric.get("id"):
        return eric, _assigned_reason(eric, "West Coast")
    return chosen, _assigned_reason(chosen)


def _assign_rep_percentile(
    row: pd.Series | dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Route by score rank vs other routable leads in the inbox (10/20/50/20 style split)."""
    score = _score(row)
    band = _percentile_band(score, config)
    gene_pct, jake_pct, general_pct, automation_pct = _distribution_pcts(config)

    bucket_key = {
        "gene": "urgent",
        "jake": "hot_warm",
        "general": "general",
        "automation": "automation",
    }

    fallthrough: list[tuple[str, str, str]] = []
    if band == "gene":
        fallthrough = [
            ("gene", "urgent", f"Top {gene_pct:.0f}% of scored leads"),
            ("jake", "hot_warm", f"Top {gene_pct:.0f}% band — Gene at weekly cap, sent to Jake"),
            ("general", "general", "Top band — Gene and Jake at cap"),
        ]
    elif band == "jake":
        fallthrough = [
            ("jake", "hot_warm", f"Next {jake_pct:.0f}% of scored leads (after top {gene_pct:.0f}%)"),
            ("general", "general", "Jake at weekly cap — general pool"),
        ]
    elif band == "general":
        fallthrough = [
            ("general", "general", f"Middle {general_pct:.0f}% of scored leads"),
        ]
    else:
        fallthrough = [
            ("automation", "automation", f"Bottom {automation_pct:.0f}% — automation / nurture"),
        ]

    for rep_key, route_bucket, _band_note in fallthrough:
        if rep_key == "general":
            rep, reason = _pick_general_rep(row, config)
            return {
                "assigned": True,
                "rep": rep,
                "route_bucket": route_bucket,
                "route_reason": reason,
                "distribution_band": band,
            }
        if rep_key == "jake":
            jake_rep, spill = _try_pick_jake(row, config)
            if jake_rep:
                return {
                    "assigned": True,
                    "rep": jake_rep,
                    "route_bucket": route_bucket,
                    "route_reason": _assigned_reason(jake_rep),
                    "distribution_band": band,
                }
            if spill and band == "jake":
                rep, _ = _pick_general_rep(row, config)
                return {
                    "assigned": True,
                    "rep": rep,
                    "route_bucket": "general",
                    "route_reason": _assigned_reason(rep, spill),
                    "distribution_band": band,
                }
            continue

        reps = reps_for_bucket(config, bucket_key[rep_key])
        rep = _first_rep_with_cap(reps)
        if rep:
            return {
                "assigned": True,
                "rep": rep,
                "route_bucket": route_bucket,
                "route_reason": _assigned_reason(rep),
                "distribution_band": band,
            }

    if band != "automation":
        rep, reason = _pick_general_rep(row, config)
        return {
            "assigned": True,
            "rep": rep,
            "route_bucket": "general",
            "route_reason": reason,
            "distribution_band": band,
        }

    auto_reps = reps_for_bucket(config, "automation")
    if auto_reps:
        rep = auto_reps[0]
        return {
            "assigned": True,
            "rep": rep,
            "route_bucket": "automation",
            "route_reason": _assigned_reason(rep),
            "distribution_band": band,
        }

    return {
        "assigned": False,
        "skipped_reason": "Automation queue not configured (add a rep with bucket automation)",
        "distribution_band": band,
    }


def _assign_rep_hybrid(
    row: pd.Series | dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Urgent red-hot → Gene first; everyone else by inbox score percentiles."""
    if is_urgent_red_hot(row, config):
        for rep in reps_for_bucket(config, "urgent"):
            if _rep_under_cap(rep):
                return {
                    "assigned": True,
                    "rep": rep,
                    "route_bucket": "urgent",
                    "route_reason": _assigned_reason(rep, "Priority lead"),
                    "distribution_band": "urgent",
                }
    return _assign_rep_percentile(row, config)


def _assign_rep_tier(
    row: pd.Series | dict[str, Any],
    config: dict[str, Any],
    *,
    ignore_prior: bool = False,
) -> dict[str, Any]:
    if is_urgent_red_hot(row, config):
        for rep in reps_for_bucket(config, "urgent"):
            if _rep_under_cap(rep):
                return {
                    "assigned": True,
                    "rep": rep,
                    "route_bucket": "urgent",
                    "route_reason": _assigned_reason(rep, "Priority lead"),
                }
        # Fall through if Gene at cap

    jake_rep, spill = _try_pick_jake(row, config)
    if jake_rep:
        return {
            "assigned": True,
            "rep": jake_rep,
            "route_bucket": "hot_warm",
            "route_reason": _assigned_reason(jake_rep),
        }

    rep, reason = _pick_general_rep(row, config)
    if spill:
        reason = _assigned_reason(rep, spill)
    return {
        "assigned": True,
        "rep": rep,
        "route_bucket": "general",
        "route_reason": reason,
    }


def assign_rep(
    row: pd.Series | dict[str, Any],
    config: dict[str, Any] | None = None,
    *,
    ignore_prior: bool = False,
    form_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Pick rep and bucket for a lead without sending email."""
    config = config or load_routing_config()
    ok, skip = should_route_lead_for_form(row, form_config)
    if not ok:
        return {"assigned": False, "skipped_reason": skip}

    get = row.get if isinstance(row, dict) else row.get
    email = _safe_str(get("Email", ""))
    if email and not ignore_prior and was_lead_routed(email):
        return {"assigned": False, "skipped_reason": "Lead already routed this cycle"}

    policy = _form_routing_policy(form_config)
    if policy == "fixed_reps" and form_config:
        return _assign_fixed_reps(row, config, form_config)

    mode = _safe_str(config.get("routing_mode", "hybrid")).lower()
    if mode == "hybrid":
        return _assign_rep_hybrid(row, config)
    if mode == "percentile":
        return _assign_rep_percentile(row, config)
    return _assign_rep_tier(row, config, ignore_prior=ignore_prior)


def route_and_notify(
    row: pd.Series | dict[str, Any],
    config: dict[str, Any] | None = None,
    *,
    ignore_prior: bool = False,
    form_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assign rep, log, optionally email, return routing result."""
    config = config or load_routing_config()
    assignment = assign_rep(row, config, ignore_prior=ignore_prior, form_config=form_config)
    if not assignment.get("assigned"):
        return assignment

    rep = assignment["rep"]
    get = row.get if isinstance(row, dict) else row.get
    lead_email = _safe_str(get("Email", ""))
    email_sent = False
    n8n_sent = False
    hubspot_synced = False
    enriched = False
    notify_error: str | None = None
    n8n_error: str | None = None

    skip_notify = assignment.get("route_bucket") == "automation" and config.get(
        "automation_skip_notify", True
    )

    # Research the lead first (bounded) so the single assignment email/SMS carries the brief.
    if not skip_notify and enrichment_enabled():
        values = enrich_before_notify(row, rep)
        if values:
            enriched = True
            merged = dict(row.items()) if not isinstance(row, dict) else dict(row)
            merged.update(values)
            row = pd.Series(merged)
    hubspot_error: str | None = None
    hubspot_result: dict[str, Any] | None = None

    if config.get("sync_hubspot_on_route", True) and hubspot_configured():
        try:
            hubspot_result = sync_contact_on_route(row, rep, assignment)
            hubspot_synced = True
        except Exception as exc:
            hubspot_error = str(exc)

    send_n8n = _form_send_to_n8n(form_config)

    if not skip_notify:
        if n8n_configured() and send_n8n:
            try:
                n8n_sent = send_n8n_assignment_notification(
                    row, rep, assignment, form_config=form_config
                )
            except Exception as exc:
                n8n_error = str(exc)
        elif not send_n8n:
            n8n_error = "Skipped n8n for this form (send_to_n8n=false)"

        if config.get("send_email_on_route", True):
            if email_configured():
                try:
                    email_sent = send_lead_assignment_email(
                        row, rep, assignment, form_config=form_config
                    )
                except Exception as exc:
                    notify_error = str(exc)
            elif not n8n_configured():
                notify_error = "No notifications configured. Set N8N_WEBHOOK_URL or email on Railway."
        elif not n8n_configured() and not email_configured():
            notify_error = "No notifications configured. Set N8N_WEBHOOK_URL or email on Railway."

    append_routing_entry(
        rep_id=_safe_str(rep.get("id")),
        rep_name=_safe_str(rep.get("name")),
        rep_email=_safe_str(rep.get("email")),
        lead_email=lead_email,
        route_bucket=assignment["route_bucket"],
        route_reason=assignment["route_reason"],
        ai_score=_score(row),
        ai_tier=_tier(row),
        email_sent=email_sent or n8n_sent,
    )

    return {
        **assignment,
        "email_sent": email_sent,
        "n8n_sent": n8n_sent,
        "hubspot_synced": hubspot_synced,
        "hubspot_result": hubspot_result,
        "hubspot_error": hubspot_error,
        "notify_error": notify_error,
        "n8n_error": n8n_error,
        "enriched": enriched,
    }
