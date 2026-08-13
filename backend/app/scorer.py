"""Blended lead scoring pipeline."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd

from .config import (
    HUBSPOT_10PT_WEIGHT,
    LLM_WEIGHT,
    ML_WEIGHT,
    RULES_WEIGHT,
)
from .features import (
    _normalize_hubspot_score,
    _safe_str,
    has_coaching_signals,
    is_high_investment_level,
    is_long_deadline,
    is_low_investment_level,
    is_near_term_deadline,
    is_new_to_wrestling,
    is_parent_icp_buyer,
    is_soft_mental_edge_inquiry,
    is_sparse_subscriber,
    is_struggling_mentally,
    is_wrestler_self_buyer,
    qualifies_icp_priority_floor,
)
from .reference_scores import apply_reference_stability
from .scoring_config import get_coaching_score_boost, get_tier_thresholds
from .llm import score_leads_with_llm_async
from .progress import ProgressCallback, emit_progress
from .train import load_model, predict_ml_scores


def score_to_tier(score: float) -> str:
    thresholds = get_tier_thresholds()
    if score >= thresholds["Hot"]:
        return "Hot"
    if score >= thresholds["Warm"]:
        return "Warm"
    if score >= thresholds["Cold"]:
        return "Cold"
    return "Unqualified"


def recommended_action(tier: str, lifecycle: str) -> str:
    if lifecycle == "Customer":
        return "Already converted — no outreach needed"
    if tier == "Hot":
        return "Priority outreach"
    if tier == "Warm":
        return "Nurture sequence"
    if tier == "Cold":
        return "Low priority follow-up"
    return "Deprioritize"


def compute_rule_adjustments(row: pd.Series) -> tuple[float, list[str], list[str]]:
    """Return rule score (0-100 scale contribution), reasons, and flags."""
    score = 55.0
    reasons: list[str] = []
    flags: list[str] = []

    lifecycle = _safe_str(row.get("Lifecycle Stage", ""))
    if lifecycle == "Customer":
        return 100.0, ["Already a customer"], []

    relationship = _safe_str(row.get("Relationship Status", ""))
    if relationship == "Ready to start now" or "ready to start" in relationship.lower():
        boost = 14 if is_parent_icp_buyer(row) else 8
        score += boost
        reasons.append("Ready to start now")

    if is_near_term_deadline(row):
        score += 12
        reasons.append("Near-term deadline")
    elif is_long_deadline(row):
        score -= 12
        reasons.append("Long-term goal (not urgent)")

    job_title = _safe_str(row.get("Job Title", ""))
    if "Parent Seeking 1-1" in job_title:
        score += 12
        reasons.append("Parent buyer (strong ICP fit)")
    elif "Wrestler Seeking 1-1" in job_title:
        score += 4
        reasons.append("Wrestler self — lower priority than parent buyers")

    job_function = _safe_str(row.get("Job function", ""))
    if is_struggling_mentally(row):
        score += 10
        reasons.append("Mental struggle (high intent)")
    elif is_soft_mental_edge_inquiry(row):
        score += 3
        reasons.append("Soft mental edge inquiry (no major issues)")
    elif "Mental Edge" in job_function:
        score += 6
        reasons.append("Seeking mental edge")

    if is_high_investment_level(row):
        score += 10
        reasons.append("High-performance investment tier")
    elif is_low_investment_level(row):
        score -= 14
        reasons.append("Budget / starter plan selected")

    if is_new_to_wrestling(row):
        score -= 8
        reasons.append("New to wrestling (0-1 year)")

    if "Coach Seeking Team Mindset Training" in job_title:
        flags.append("Coach/team track — verify product fit")
        score -= 5

    if not _safe_str(row.get("Email", "")):
        score -= 10
        flags.append("Missing email")
    if not _safe_str(row.get("Phone Number", "")):
        score -= 5
        flags.append("Missing phone")

    lead_status = _safe_str(row.get("Lead Status", ""))
    if lead_status == "Unqualified":
        score = min(score, 30.0)
        flags.append("HubSpot marked Unqualified")

    if lifecycle == "Subscriber" and has_coaching_signals(row):
        score += 10
        reasons.append("Existing subscriber with coaching signals")
    elif is_sparse_subscriber(row):
        flags.append("Book/content subscriber — no coaching form data")

    return max(0.0, min(100.0, score)), reasons, flags


def blend_scores(
    ml_score: float,
    llm_score: float,
    hubspot_score: float | None,
    rule_score: float,
) -> float:
    """Weighted blend on 0-100 scale."""
    hubspot_component = hubspot_score if hubspot_score is not None and not np.isnan(hubspot_score) else rule_score

    total_weight = ML_WEIGHT + LLM_WEIGHT + RULES_WEIGHT
    if hubspot_score is not None and not np.isnan(hubspot_score):
        total_weight += HUBSPOT_10PT_WEIGHT
        return (
            ML_WEIGHT * ml_score
            + LLM_WEIGHT * llm_score
            + HUBSPOT_10PT_WEIGHT * hubspot_component
            + RULES_WEIGHT * rule_score
        )

    # Redistribute hubspot weight to ML/LLM when 10pt score missing
    ml_w = ML_WEIGHT + HUBSPOT_10PT_WEIGHT / 2
    llm_w = LLM_WEIGHT + HUBSPOT_10PT_WEIGHT / 2
    return ml_w * ml_score + llm_w * llm_score + RULES_WEIGHT * rule_score


def apply_icp_priority_floor(
    final: float,
    row: pd.Series,
    llm_score: float,
) -> tuple[float, str | None]:
    """Raise score to Hot threshold when text + ICP + readiness are strong."""
    hot_floor = float(get_tier_thresholds()["Hot"])
    if final >= hot_floor or not qualifies_icp_priority_floor(row, llm_score):
        return final, None
    return hot_floor, "ICP priority: high-intent coaching lead (text + readiness)"


def score_dataframe(
    df: pd.DataFrame,
    use_llm: bool = True,
    max_llm_rows: int | None = None,
) -> pd.DataFrame:
    """Sync wrapper — use score_dataframe_async from async endpoints."""
    import asyncio

    return asyncio.run(score_dataframe_async(df, use_llm=use_llm, max_llm_rows=max_llm_rows))


async def score_dataframe_async(
    df: pd.DataFrame,
    use_llm: bool = True,
    max_llm_rows: int | None = None,
    on_progress: ProgressCallback | None = None,
) -> pd.DataFrame:
    """Score all leads and return enriched DataFrame."""
    if df.empty:
        return df.copy()

    total = len(df)

    emit_progress(on_progress, "ml", "Running ML model", 0, 1, use_llm)
    model = load_model()
    _, ml_scores = predict_ml_scores(df, model=model)
    emit_progress(on_progress, "ml", "ML model complete", 1, 1, use_llm)

    llm_label = "DeepSeek text scoring" if use_llm else "Heuristic text scoring"
    emit_progress(on_progress, "llm", llm_label, 0, total, use_llm)

    def llm_row_progress(done: int, llm_total: int) -> None:
        emit_progress(on_progress, "llm", llm_label, done, llm_total, use_llm)

    llm_results = await score_leads_with_llm_async(
        df,
        use_llm=use_llm,
        max_rows=max_llm_rows,
        on_row_complete=llm_row_progress,
    )

    enriched = df.copy()
    ai_scores: list[float] = []
    ai_tiers: list[str] = []
    ai_reasons: list[str] = []
    ai_actions: list[str] = []
    ml_cols: list[float] = []
    llm_cols: list[float] = []
    rule_cols: list[float] = []

    emit_progress(on_progress, "blending", "Blending scores", 0, total, use_llm)

    for idx, (_, row) in enumerate(df.iterrows()):
        lifecycle = _safe_str(row.get("Lifecycle Stage", ""))
        llm = llm_results[idx] if idx < len(llm_results) else {"score": 50, "reasons": [], "red_flags": []}

        if lifecycle == "Customer":
            final = 100.0
            tier = "Hot"
            reasons = ["Already converted customer"]
            action = recommended_action(tier, lifecycle)
            rule_score = 100.0
        else:
            hubspot_raw = row.get("10-Point Lead Score")
            hubspot_norm = _normalize_hubspot_score(hubspot_raw)
            hubspot_val = None if np.isnan(hubspot_norm) else hubspot_norm

            rule_score, rule_reasons, rule_flags = compute_rule_adjustments(row)

            final = blend_scores(
                ml_score=float(ml_scores[idx]),
                llm_score=float(llm["score"]),
                hubspot_score=hubspot_val,
                rule_score=rule_score,
            )
            boost = get_coaching_score_boost()
            if boost > 0 and has_coaching_signals(row):
                final += boost
            final = max(0.0, min(100.0, final))

            boost_note: str | None = None
            if boost > 0 and has_coaching_signals(row):
                boost_note = f"Coaching intent boost +{boost:.0f}"

            cap_reason: str | None = None
            final, stability_reason = apply_reference_stability(final, row)
            if stability_reason:
                cap_reason = stability_reason

            final, icp_reason = apply_icp_priority_floor(final, row, float(llm["score"]))
            if icp_reason:
                cap_reason = icp_reason

            tier = score_to_tier(final)

            reason_parts = [
                f"ML: {ml_scores[idx]:.0f}/100",
                f"Text: {llm['score']:.0f}/100",
            ]
            if hubspot_val is not None:
                reason_parts.append(f"HubSpot 10pt: {hubspot_val:.0f}/100")
            reason_parts.extend(llm.get("reasons", [])[:2])
            reason_parts.extend(rule_reasons[:2])
            if boost_note:
                reason_parts.append(boost_note)
            if cap_reason:
                reason_parts.append(cap_reason)
            flags = llm.get("red_flags", []) + rule_flags
            if flags:
                reason_parts.append("Flags: " + "; ".join(flags[:2]))
            reasons = reason_parts
            action = recommended_action(tier, lifecycle)

        ai_scores.append(round(final, 1))
        ai_tiers.append(tier)
        ai_reasons.append(" | ".join(reasons))
        ai_actions.append(action)
        ml_cols.append(round(float(ml_scores[idx]), 1))
        llm_cols.append(round(float(llm["score"]), 1))
        rule_cols.append(round(float(rule_score), 1))

        if (idx + 1) % 25 == 0 or idx + 1 == total:
            emit_progress(on_progress, "blending", "Blending scores", idx + 1, total, use_llm)

    enriched["AI Score"] = ai_scores
    enriched["AI Tier"] = ai_tiers
    enriched["AI Reasons"] = ai_reasons
    enriched["Recommended Action"] = ai_actions
    enriched["ML Score"] = ml_cols
    enriched["LLM Score"] = llm_cols
    enriched["Rule Score"] = rule_cols

    return enriched.sort_values("AI Score", ascending=False).reset_index(drop=True)


def metrics_summary(scored: pd.DataFrame) -> dict[str, Any]:
    tier_counts = scored["AI Tier"].value_counts().to_dict() if "AI Tier" in scored.columns else {}
    return {
        "total_leads": len(scored),
        "tier_counts": tier_counts,
        "average_score": round(float(scored["AI Score"].mean()), 1) if len(scored) else 0,
    }
