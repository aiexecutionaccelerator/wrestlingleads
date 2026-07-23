"""Feature engineering for tabular ML and text scoring."""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .config import CATEGORICAL_COLUMNS

URGENT_DEADLINE_KEYWORDS = (
    "now",
    "asap",
    "immediately",
    "this week",
    "next week",
    "next month",
    "urgent",
)
COACHING_SOURCE_MARKERS = ("wufoo", "1-1")
INCOMPLETE_PROFILE_MAX_SCORE = 35.0

CORE_ICP_JOB_TITLES = (
    "Parent Seeking 1-1 Coaching for Child",
    "Wrestler Seeking 1-1 Coaching for Myself",
)
MENTAL_STRUGGLE_JOB_FUNCTIONS = (
    "Struggling mentally",
    "Mental Edge",
)

LOW_INVESTMENT_MARKERS = (
    "getting started",
    "lower-cost",
    "limited-contact",
    "foundational",
    "limited contact",
    "lower cost",
)


PLACEHOLDER_TEXT_VALUES = frozenset({"n/a", "na", "none", "no", "nothing", "-", "."})


def _safe_str(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).strip()


def _has_value(value: object) -> bool:
    return bool(_safe_str(value))


def _meaningful_text(value: object) -> str:
    """Free-text value with 'N/a'-style placeholders treated as empty."""
    text = _safe_str(value)
    return "" if text.lower() in PLACEHOLDER_TEXT_VALUES else text


def _normalize_hubspot_score(value: object) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return np.nan
    if score < 0:
        return 0.0
    if score > 10:
        return 100.0
    return score * 10.0


def _deadline_urgency(value: object) -> int:
    text = _safe_str(value).lower()
    if not text:
        return 0
    return int(any(keyword in text for keyword in URGENT_DEADLINE_KEYWORDS))


def build_tabular_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create engineered tabular features used by the ML model."""
    features = pd.DataFrame(index=df.index)

    for col in CATEGORICAL_COLUMNS:
        if col in df.columns:
            features[col] = df[col].apply(_safe_str).replace("", "Unknown")
        else:
            features[col] = "Unknown"

    features["hubspot_10pt_normalized"] = (
        df["10-Point Lead Score"].apply(_normalize_hubspot_score)
        if "10-Point Lead Score" in df.columns
        else np.nan
    )
    features["deadline_urgency"] = (
        df["Deadline for Goal"].apply(_deadline_urgency)
        if "Deadline for Goal" in df.columns
        else 0
    )
    features["has_email"] = df["Email"].apply(_has_value).astype(int) if "Email" in df.columns else 0
    features["has_phone"] = (
        df["Phone Number"].apply(_has_value).astype(int) if "Phone Number" in df.columns else 0
    )
    features["has_message"] = (
        df["Message"].apply(lambda v: bool(_meaningful_text(v))).astype(int) if "Message" in df.columns else 0
    )
    features["message_length"] = (
        df["Message"].apply(lambda v: len(_meaningful_text(v))) if "Message" in df.columns else 0
    )

    return features


def has_name(row: pd.Series | dict) -> bool:
    if isinstance(row, dict):
        return bool(_safe_str(row.get("First Name")) or _safe_str(row.get("Last Name")))
    return bool(_safe_str(row.get("First Name", "")) or _safe_str(row.get("Last Name", "")))


def has_coaching_signals(row: pd.Series | dict) -> bool:
    """True when lead has data indicating 1-on-1 coaching interest (not email-only)."""
    get = row.get if isinstance(row, dict) else row.get
    if _meaningful_text(get("Message", "")):
        return True
    if _has_value(get("Job Title", "")):
        return True
    if _has_value(get("Job function", "")):
        return True
    if _has_value(get("Investment Level", "")):
        return True
    if _has_value(get("Wrestler's Goal", "")):
        return True
    source = _safe_str(get("Source", "")).lower()
    return any(marker in source for marker in COACHING_SOURCE_MARKERS)


def is_sparse_subscriber(row: pd.Series | dict) -> bool:
    lifecycle = _safe_str(row.get("Lifecycle Stage", "") if isinstance(row, dict) else row.get("Lifecycle Stage", ""))
    return lifecycle == "Subscriber" and not has_coaching_signals(row)


def is_incomplete_profile(row: pd.Series | dict) -> bool:
    """Email-only or minimal HubSpot record — not enough to qualify for coaching outreach."""
    return not has_coaching_signals(row) and not has_name(row)


def is_near_term_deadline(row: pd.Series | dict) -> bool:
    get = row.get if isinstance(row, dict) else row.get
    deadline = _safe_str(get("Deadline for Goal", "")).lower()
    return any(keyword in deadline for keyword in URGENT_DEADLINE_KEYWORDS)


def is_long_deadline(row: pd.Series | dict) -> bool:
    """Multi-year or vague far-future goals — not urgent."""
    get = row.get if isinstance(row, dict) else row.get
    deadline = _safe_str(get("Deadline for Goal", "")).lower()
    if not deadline:
        return False
    if is_near_term_deadline(row):
        return False
    if re.search(r"\b\d+\s*\+?\s*year", deadline):
        return True
    if " years" in deadline or deadline in ("year", "years"):
        return True
    return False


def is_low_investment_level(row: pd.Series | dict) -> bool:
    get = row.get if isinstance(row, dict) else row.get
    investment = _safe_str(get("Investment Level", "")).lower()
    return any(marker in investment for marker in LOW_INVESTMENT_MARKERS)


def is_high_investment_level(row: pd.Series | dict) -> bool:
    get = row.get if isinstance(row, dict) else row.get
    investment = _safe_str(get("Investment Level", "")).lower()
    if is_low_investment_level(row):
        return False
    return "high-performance" in investment or (
        "recommended" in investment and "getting started" not in investment
    )


def is_parent_icp_buyer(row: pd.Series | dict) -> bool:
    get = row.get if isinstance(row, dict) else row.get
    return "Parent Seeking 1-1" in _safe_str(get("Job Title", ""))


def is_wrestler_self_buyer(row: pd.Series | dict) -> bool:
    get = row.get if isinstance(row, dict) else row.get
    return "Wrestler Seeking 1-1" in _safe_str(get("Job Title", ""))


def is_struggling_mentally(row: pd.Series | dict) -> bool:
    get = row.get if isinstance(row, dict) else row.get
    return "Struggling mentally" in _safe_str(get("Job function", ""))


def is_soft_mental_edge_inquiry(row: pd.Series | dict) -> bool:
    get = row.get if isinstance(row, dict) else row.get
    inquiry = _safe_str(get("Job function", "")).lower()
    if "mental edge" not in inquiry:
        return False
    return any(
        phrase in inquiry
        for phrase in ("no major", "no issue", "no major issue", "no major issues")
    )


def is_ready_for_coaching(row: pd.Series | dict) -> bool:
    get = row.get if isinstance(row, dict) else row.get
    relationship = _safe_str(get("Relationship Status", "")).lower()
    if "ready to start" in relationship:
        return True
    return is_near_term_deadline(row)


def is_core_icp_buyer(row: pd.Series | dict) -> bool:
    get = row.get if isinstance(row, dict) else row.get
    job_title = _safe_str(get("Job Title", ""))
    if any(title in job_title for title in CORE_ICP_JOB_TITLES):
        return True
    job_function = _safe_str(get("Job function", ""))
    return any(marker in job_function for marker in MENTAL_STRUGGLE_JOB_FUNCTIONS)


def qualifies_icp_priority_floor(row: pd.Series | dict, llm_score: float) -> bool:
    """High-intent 1-on-1 coaching lead — ML should not demote below Priority."""
    from .scoring_config import get_icp_llm_min

    if llm_score < get_icp_llm_min():
        return False
    if is_sparse_subscriber(row):
        return False
    if not has_coaching_signals(row):
        return False
    if not is_ready_for_coaching(row):
        return False
    if is_low_investment_level(row):
        return False
    if is_long_deadline(row):
        return False

    get = row.get if isinstance(row, dict) else row.get
    if "Coach Seeking Team Mindset Training" in _safe_str(get("Job Title", "")):
        return False
    if _safe_str(get("Lead Status", "")) == "Unqualified":
        return False

    if is_parent_icp_buyer(row):
        if is_soft_mental_edge_inquiry(row) and not is_struggling_mentally(row):
            return False
        return True

    if is_wrestler_self_buyer(row):
        return (
            is_struggling_mentally(row)
            and is_near_term_deadline(row)
            and is_high_investment_level(row)
        )

    return is_struggling_mentally(row) and is_near_term_deadline(row)


def empty_profile_text_result() -> dict[str, Any]:
    return {
        "score": 15.0,
        "reasons": ["Insufficient coaching form data"],
        "red_flags": ["Email-only record — likely book/content subscriber"],
        "source": "empty_profile",
    }


def build_text_bundle(row: pd.Series) -> str:
    """Concatenate text fields for LLM scoring."""
    parts: list[str] = []
    field_labels = [
        ("Message", "Message"),
        ("Job function", "Job function"),
        ("Wrestler's Goal", "Goal"),
        ("Membership Notes", "Notes"),
        ("Job Title", "Buyer type"),
        ("Relationship Status", "Readiness"),
        ("Deadline for Goal", "Deadline"),
        ("Investment Level", "Investment level"),
        ("Years experience", "Experience"),
        ("Wrestler's Grade", "Grade"),
    ]
    for col, label in field_labels:
        value = _safe_str(row.get(col, ""))
        if col == "Message":
            value = _meaningful_text(value)
        if value:
            parts.append(f"{label}: {value}")
    return "\n".join(parts)


def heuristic_text_score(row: pd.Series) -> tuple[float, list[str], list[str]]:
    """Fallback text score when LLM is unavailable."""
    if not build_text_bundle(row).strip():
        result = empty_profile_text_result()
        return result["score"], result["reasons"], result["red_flags"]

    # Skip the investment-plan line: its marketing copy ("mindset skills") would
    # false-trigger the mental-pain patterns; investment is scored structurally below.
    text = "\n".join(
        line
        for line in build_text_bundle(row).lower().splitlines()
        if not line.startswith("investment level:")
    )
    score = 42.0
    reasons: list[str] = []
    red_flags: list[str] = []

    positive_patterns = [
        (r"\b(struggl|nerv|anxi|choke|confidence|mental|mindset)\b", 8, "Mental performance pain described"),
        (r"\b(state|national|tournament|placer|qualifier)\b", 6, "Competitive experience mentioned"),
        (r"\b(ready to start|start now)\b", 5, "Readiness language present"),
    ]
    negative_patterns = [
        (r"\b(first year|new to wrestling|no team)\b", -5, "Very early / low commitment signal"),
        (r"\b(bjj|baseball|soccer)\b", -3, "Non-wrestling primary sport mentioned"),
    ]

    for pattern, delta, reason in positive_patterns:
        if re.search(pattern, text):
            score += delta
            reasons.append(reason)

    for pattern, delta, reason in negative_patterns:
        if re.search(pattern, text):
            score += delta
            red_flags.append(reason)

    job_title = _safe_str(row.get("Job Title", ""))
    if "Parent Seeking" in job_title:
        score += 10
        reasons.append("Parent buyer (strong fit historically)")
    elif "Wrestler Seeking" in job_title:
        score += 2
        reasons.append("Wrestler self-signup (weaker ICP than parent)")
    elif "Coach Seeking" in job_title:
        score += 3
        reasons.append("Coach buyer (different product track)")

    job_function = _safe_str(row.get("Job function", ""))
    if "Struggling mentally" in job_function:
        score += 10
        reasons.append("Explicit mental struggle")
    elif is_soft_mental_edge_inquiry(row):
        score += 2
        reasons.append("Soft mental edge inquiry")
    elif "Mental Edge" in job_function:
        score += 5
        reasons.append("Seeking mental edge")

    if is_low_investment_level(row):
        score -= 12
        red_flags.append("Starter / budget plan selected")
    elif is_high_investment_level(row):
        score += 6
        reasons.append("High-performance investment tier")

    if is_long_deadline(row):
        score -= 8
        red_flags.append("Long-term deadline — not urgent")

    if not text.strip():
        score = 15.0
        red_flags.append("No message or context provided")

    score = max(0.0, min(100.0, score))
    if not reasons:
        reasons.append("Heuristic text analysis (no DeepSeek API key configured)")
    return score, reasons[:3], red_flags[:3]


class NamedFeatureFrame(BaseEstimator, TransformerMixin):
    """Give preprocessed numeric arrays stable column names for LightGBM."""

    def fit(self, X, y=None):
        sample = X.to_numpy() if hasattr(X, "to_numpy") else np.asarray(X)
        self.columns_ = [f"feature_{i}" for i in range(sample.shape[1])]
        return self

    def transform(self, X):
        values = X.to_numpy() if hasattr(X, "to_numpy") else np.asarray(X)
        return pd.DataFrame(values, columns=self.columns_)


def create_preprocessing_pipeline(feature_columns: list[str]) -> ColumnTransformer:
    categorical_cols = [c for c in CATEGORICAL_COLUMNS if c in feature_columns]
    numeric_cols = [
        c
        for c in feature_columns
        if c not in categorical_cols
    ]

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    transformers = []
    if categorical_cols:
        transformers.append(("cat", categorical_transformer, categorical_cols))
    if numeric_cols:
        transformers.append(("num", numeric_transformer, numeric_cols))

    preprocessor = ColumnTransformer(transformers=transformers)
    return preprocessor
