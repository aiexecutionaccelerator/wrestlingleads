"""Build and refresh all AI assets from qualified training data."""

from __future__ import annotations

import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from .config import (
    FEW_SHOT_PATH,
    HUBSPOT_10PT_WEIGHT,
    LLM_WEIGHT,
    METRICS_PATH,
    ML_WEIGHT,
    MODEL_PATH,
    MODELS_DIR,
    RULES_WEIGHT,
    TIER_THRESHOLDS,
)
from .train import train_model
from .reference_scores import save_reference

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
CONFIG_DIR = BASE_DIR / "config"
QUALIFIED_PATH = DATA_DIR / "qualified.xlsx"


def _safe_str(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).strip()


def load_qualified(path: Path | None = None) -> pd.DataFrame:
    path = path or QUALIFIED_PATH
    if not path.exists():
        path = BASE_DIR / "qualified.xlsx"
    if not path.exists():
        raise FileNotFoundError("qualified.xlsx not found in data/ or project root.")
    return pd.read_excel(path, engine="openpyxl")


def build_few_shot_examples(df: pd.DataFrame, n: int = 8) -> list[dict]:
    """Pick strong Hot/Customer examples with real message content for DeepSeek."""
    candidates = df.copy()

    if "AI Tier" in candidates.columns:
        hot = candidates[candidates["AI Tier"].astype(str).str.strip() == "Hot"]
        if not hot.empty:
            candidates = hot

    lifecycle = candidates.get("Lifecycle Stage", pd.Series(dtype=str)).astype(str).str.strip()
    customers = candidates[lifecycle.isin(["Customer", "Subscriber"])]
    if len(customers) >= 3:
        pool = customers
    else:
        pool = candidates

    pool = pool.copy()
    pool["_msg_len"] = pool.get("Message", pd.Series(dtype=str)).apply(lambda v: len(_safe_str(v)))
    pool = pool.sort_values("_msg_len", ascending=False)

    examples: list[dict] = []
    seen_messages: set[str] = set()
    for _, row in pool.iterrows():
        message = _safe_str(row.get("Message", ""))[:500]
        if len(message) < 40:
            continue
        key = message[:80]
        if key in seen_messages:
            continue
        seen_messages.add(key)

        ai_score = row.get("AI Score", 90)
        try:
            score = float(ai_score)
        except (TypeError, ValueError):
            score = 90.0

        examples.append(
            {
                "name": f"{row.get('First Name', '')} {row.get('Last Name', '')}".strip(),
                "job_title": _safe_str(row.get("Job Title", "")),
                "job_function": _safe_str(row.get("Job function", "")),
                "message": message,
                "lifecycle": _safe_str(row.get("Lifecycle Stage", "Customer")),
                "ai_tier": _safe_str(row.get("AI Tier", "Hot")),
                "score": round(score, 1),
                "reasons": [
                    _safe_str(row.get("Recommended Action", "")) or "Strong fit for 1-on-1 mental coaching",
                    "Reference example from qualified lead dataset",
                ],
            }
        )
        if len(examples) >= n:
            break

    return examples


def build_tier_calibration(df: pd.DataFrame) -> dict:
    tier_col = df.get("AI Tier")
    if tier_col is None:
        return {}

    calibration: dict = {}
    for tier in ["Hot", "Warm", "Cold", "Unqualified"]:
        subset = df[tier_col.astype(str).str.strip() == tier]
        if subset.empty:
            continue
        calibration[tier] = {
            "count": int(len(subset)),
            "avg_ai_score": round(float(subset["AI Score"].mean()), 1) if "AI Score" in subset else None,
            "avg_ml_score": round(float(subset["ML Score"].mean()), 1) if "ML Score" in subset else None,
            "avg_llm_score": round(float(subset["LLM Score"].mean()), 1) if "LLM Score" in subset else None,
            "top_job_titles": subset["Job Title"].value_counts().head(5).to_dict() if "Job Title" in subset else {},
            "top_lifecycle": subset["Lifecycle Stage"].value_counts().to_dict()
            if "Lifecycle Stage" in subset
            else {},
        }
    return calibration


def build_scoring_config() -> dict:
    return {
        "weights": {
            "ml": ML_WEIGHT,
            "llm": LLM_WEIGHT,
            "hubspot_10pt": HUBSPOT_10PT_WEIGHT,
            "rules": RULES_WEIGHT,
        },
        "tiers": TIER_THRESHOLDS,
        "llm_provider": "deepseek",
        "llm_model_env": "DEEPSEEK_MODEL",
        "positive_lifecycle_stages": ["Customer", "Subscriber"],
    }


def build_hubspot_config() -> dict:
    return {
        "contact_properties": {
            "ai_score": {
                "label": "AI Score",
                "type": "number",
                "description": "Blended qualification score 0-100",
            },
            "ai_tier": {
                "label": "AI Tier",
                "type": "enumeration",
                "options": ["Hot", "Warm", "Cold", "Unqualified"],
                "description": "Lead priority tier from AI qualifier",
            },
            "ai_reasons": {
                "label": "AI Reasons",
                "type": "string",
                "description": "Why the lead received this score",
            },
            "recommended_action": {
                "label": "Recommended Action",
                "type": "string",
                "description": "Suggested next step for sales",
            },
            "ml_score": {"label": "ML Score", "type": "number"},
            "llm_score": {"label": "LLM Score", "type": "number"},
            "lw_assigned_rep": {
                "label": "LW Assigned Rep",
                "type": "string",
                "description": "Sales rep from LeadsWrestling routing",
            },
            "lw_route_reason": {
                "label": "LW Route Reason",
                "type": "string",
            },
            "lw_assigned_at": {
                "label": "LW Assigned At",
                "type": "string",
            },
        },
        "source_to_hubspot_field_map": {
            "AI Score": "ai_score",
            "AI Tier": "ai_tier",
            "AI Reasons": "ai_reasons",
            "Recommended Action": "recommended_action",
            "ML Score": "ml_score",
            "LLM Score": "llm_score",
            "First Name": "firstname",
            "Last Name": "lastname",
            "Email": "email",
            "Phone Number": "phone",
            "Message": "message",
            "Job Title": "jobtitle",
            "Lifecycle Stage": "lifecyclestage",
            "Lead Status": "hs_lead_status",
        },
    }


def build_wufoo_config() -> dict:
    return {
        "note": "Replace FieldN keys with your actual Wufoo field IDs from form settings.",
        "wufoo_to_qualifier_map": {
            "Field1": "First Name",
            "Field2": "Last Name",
            "Field3": "Email",
            "Field4": "Phone Number",
            "Field5": "Message",
            "Field6": "Job Title",
            "Field7": "Job function",
            "Field8": "Relationship Status",
            "Field9": "Investment Level",
            "Field10": "Years experience",
            "Field11": "Wrestler's Grade",
            "Field12": "Wrestler's Goal",
            "Field13": "Deadline for Goal",
            "Field14": "State/Region",
        },
        "qualifier_to_hubspot_map": build_hubspot_config()["source_to_hubspot_field_map"],
    }


def build_ai_manifest(df: pd.DataFrame, metrics: dict, few_shot: list[dict], reference_path: Path) -> dict:
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "qualified_rows": len(df),
        "artifacts": {
            "qualified_data": str(QUALIFIED_PATH.relative_to(BASE_DIR)),
            "ml_model": str(MODEL_PATH.relative_to(BASE_DIR)),
            "few_shot_examples": str(FEW_SHOT_PATH.relative_to(BASE_DIR)),
            "metrics": str(METRICS_PATH.relative_to(BASE_DIR)),
            "scoring_config": "models/scoring_config.json",
            "tier_calibration": "models/tier_calibration.json",
            "reference_scores": str(reference_path.relative_to(BASE_DIR)).replace("\\", "/"),
            "hubspot_config": "config/hubspot_field_map.json",
            "wufoo_config": "config/wufoo_field_map.json",
        },
        "tier_counts": df["AI Tier"].value_counts().to_dict() if "AI Tier" in df.columns else {},
        "few_shot_count": len(few_shot),
        "model_metrics": metrics,
    }


def build_all(qualified_path: Path | None = None) -> dict:
    """Generate every AI asset file from qualified.xlsx."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    source = qualified_path or BASE_DIR / "qualified.xlsx"
    if source.exists() and source.resolve() != QUALIFIED_PATH.resolve():
        shutil.copy2(source, QUALIFIED_PATH)

    df = load_qualified(QUALIFIED_PATH)

    few_shot = build_few_shot_examples(df)
    FEW_SHOT_PATH.write_text(json.dumps(few_shot, indent=2), encoding="utf-8")

    metrics = train_model()

    scoring_config = build_scoring_config()
    (MODELS_DIR / "scoring_config.json").write_text(
        json.dumps(scoring_config, indent=2), encoding="utf-8"
    )

    tier_calibration = build_tier_calibration(df)
    (MODELS_DIR / "tier_calibration.json").write_text(
        json.dumps(tier_calibration, indent=2), encoding="utf-8"
    )

    reference_path = save_reference(df)

    # Bootstrap templates only — never overwrite live field maps (they hold real Wufoo field IDs).
    hubspot_path = CONFIG_DIR / "hubspot_field_map.json"
    if not hubspot_path.exists():
        hubspot_path.write_text(json.dumps(build_hubspot_config(), indent=2), encoding="utf-8")

    wufoo_path = CONFIG_DIR / "wufoo_field_map.json"
    if not wufoo_path.exists():
        wufoo_path.write_text(json.dumps(build_wufoo_config(), indent=2), encoding="utf-8")

    manifest = build_ai_manifest(df, metrics, few_shot, reference_path)
    (MODELS_DIR / "ai_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return manifest


if __name__ == "__main__":
    manifest = build_all()
    print(json.dumps(manifest, indent=2))
