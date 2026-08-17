"""Persistent cache of scored leads for dashboard and instant export."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from .config import BASE_DIR, DATA_DIR
from .lead_form_fields import FORM_COLUMN_NAMES
from .scorer import metrics_summary

CACHE_PARQUET = DATA_DIR / "scored_cache.parquet"
CACHE_META = DATA_DIR / "scored_cache_meta.json"
WEBHOOK_RECENT_LOG = DATA_DIR / "webhook_recent_log.json"
BASELINE_PARQUET = DATA_DIR / "baseline_qualified.parquet"
BASELINE_META = DATA_DIR / "baseline_qualified_meta.json"
QUALIFIED_FALLBACK = DATA_DIR / "qualified.xlsx"
ROOT_QUALIFIED = BASE_DIR / "qualified.xlsx"

DISPLAY_COLUMNS = [
    "Record ID",
    *FORM_COLUMN_NAMES,
    "Lifecycle Stage",
    "Lead Status",
    "AI Score",
    "AI Tier",
    "ML Score",
    "LLM Score",
    "Recommended Action",
    "AI Reasons",
    "Create Date",
    "Assigned Rep",
    "Assigned Email",
    "Route Bucket",
    "Routed At",
]

# Dev/integration test rows — never show in Recent or keep in cache after restart
_SYNTHETIC_TEST_EMAILS = frozenset(
    {
        "webhook-test-leads@example.com",
        "wufoo-form-test@example.com",
    }
)
_SYNTHETIC_TEST_RECORD_IDS = frozenset({"99999901", "99999902"})
_MAX_WEBHOOK_LOG = 200


def _safe_str(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).strip()


def _norm_email(value: object) -> str:
    return _safe_str(value).lower()


def is_synthetic_test_lead(row: pd.Series | dict[str, Any]) -> bool:
    get = row.get if isinstance(row, dict) else row.get
    email = _norm_email(get("Email", ""))
    if email in _SYNTHETIC_TEST_EMAILS or email.endswith("@example.com"):
        return True
    record_id = _safe_str(get("Record ID", ""))
    return record_id in _SYNTHETIC_TEST_RECORD_IDS


def _parse_create_date(series: pd.Series) -> pd.Series:
    if series is None or series.empty:
        return pd.Series(dtype="datetime64[ns]")
    return pd.to_datetime(series.astype(str).replace("", pd.NA), errors="coerce")


def _sort_dashboard_newest(df: pd.DataFrame) -> pd.DataFrame:
    """Newest leads first — Wufoo appends must be visible on page 1."""
    if df.empty:
        return df
    out = df.copy()
    if "Create Date" in out.columns:
        out["_sort_ts"] = _parse_create_date(out["Create Date"])
    else:
        out["_sort_ts"] = pd.NaT
    # Fallback: last rows in cache are usually the most recently appended
    out["_append_order"] = range(len(out))
    out = out.sort_values(
        ["_sort_ts", "_append_order"],
        ascending=[False, False],
        na_position="last",
    )
    return out.drop(columns=["_sort_ts", "_append_order"], errors="ignore")


def _is_customer(row: pd.Series | dict[str, Any]) -> bool:
    if isinstance(row, dict):
        lifecycle = _safe_str(row.get("Lifecycle Stage", ""))
    else:
        lifecycle = _safe_str(row.get("Lifecycle Stage", ""))
    return lifecycle == "Customer"


def _read_webhook_log() -> list[dict[str, Any]]:
    if not WEBHOOK_RECENT_LOG.exists():
        return []
    try:
        data = json.loads(WEBHOOK_RECENT_LOG.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError):
        return []


def _write_webhook_log(entries: list[dict[str, Any]]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    WEBHOOK_RECENT_LOG.write_text(json.dumps(entries[:_MAX_WEBHOOK_LOG], indent=2), encoding="utf-8")


def log_webhook_lead(record_id: str = "", email: str = "") -> None:
    """Track Wufoo webhook ingests for the dashboard 'Just came in' strip only."""
    email_key = _norm_email(email)
    record_key = _safe_str(record_id)
    if not email_key and not record_key:
        return

    entries = _read_webhook_log()
    entries = [
        e
        for e in entries
        if (email_key and _norm_email(e.get("email", "")) != email_key)
        and (not record_key or _safe_str(e.get("record_id", "")) != record_key)
    ]
    entries.insert(
        0,
        {
            "record_id": record_key,
            "email": email_key,
            "at": datetime.now(UTC).isoformat(),
        },
    )
    _write_webhook_log(entries)


class ScoredLeadsStore:
    def __init__(self) -> None:
        self._df: pd.DataFrame | None = None
        self._meta: dict[str, Any] = {}
        self._baseline_df: pd.DataFrame | None = None
        self._baseline_meta: dict[str, Any] = {}

    @property
    def loaded(self) -> bool:
        return self._df is not None and not self._df.empty

    @property
    def baseline_loaded(self) -> bool:
        return self._baseline_df is not None and not self._baseline_df.empty

    def load_on_startup(self) -> None:
        if CACHE_PARQUET.exists():
            self._df = pd.read_parquet(CACHE_PARQUET)
            self._meta = self._read_meta()
            self._ensure_routing_columns()
        else:
            for path in (QUALIFIED_FALLBACK, ROOT_QUALIFIED):
                if path.exists() and self._load_qualified_file(path):
                    break

        if BASELINE_PARQUET.exists():
            self._baseline_df = pd.read_parquet(BASELINE_PARQUET)
            self._baseline_meta = self._read_baseline_meta()

        self.purge_synthetic_test_leads()
        self._ensure_routing_columns()

    def purge_synthetic_test_leads(self) -> int:
        """Remove integration-test rows from cache (e.g. @example.com webhook probes)."""
        if self._df is None or self._df.empty:
            return 0
        mask = self._df.apply(is_synthetic_test_lead, axis=1)
        removed = int(mask.sum())
        if removed:
            self._df = self._df[~mask].reset_index(drop=True)
            self._meta["row_count"] = len(self._df)
            self._persist()
        return removed

    def _read_meta(self) -> dict[str, Any]:
        if CACHE_META.exists():
            return json.loads(CACHE_META.read_text(encoding="utf-8"))
        return {}

    def _read_baseline_meta(self) -> dict[str, Any]:
        if BASELINE_META.exists():
            return json.loads(BASELINE_META.read_text(encoding="utf-8"))
        return {}

    def _write_baseline_meta(self) -> None:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        BASELINE_META.write_text(json.dumps(self._baseline_meta, indent=2), encoding="utf-8")

    def _write_meta(self) -> None:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        CACHE_META.write_text(json.dumps(self._meta, indent=2), encoding="utf-8")

    def _persist(self) -> None:
        assert self._df is not None
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        self._df.to_parquet(CACHE_PARQUET, index=False)
        self._write_meta()

    def _ensure_routing_columns(self) -> None:
        if self._df is None:
            return
        for col in ("Assigned Rep", "Assigned Email", "Route Bucket", "Route Reason", "Routed At"):
            if col not in self._df.columns:
                self._df[col] = ""

    def find_lead_index(self, email: str = "", record_id: str = "") -> int | None:
        if not self.loaded or self._df is None:
            return None
        if email:
            idx = self._find_email_index(email)
            if idx is not None:
                return idx
        if record_id and "Record ID" in self._df.columns:
            matches = self._df[self._df["Record ID"].astype(str).str.strip() == str(record_id).strip()]
            if not matches.empty:
                return int(matches.index[0])
        return None

    def get_row_at(self, idx: int) -> pd.Series:
        assert self._df is not None
        return self._df.iloc[idx]

    def apply_routing_result(self, email: str, result: dict[str, Any]) -> None:
        if not result.get("assigned") or not result.get("rep"):
            return
        idx = self._find_email_index(email)
        if idx is None or self._df is None:
            return
        rep = result["rep"]
        self._ensure_routing_columns()
        self._df.at[idx, "Assigned Rep"] = _safe_str(rep.get("name"))
        self._df.at[idx, "Assigned Email"] = _safe_str(rep.get("email"))
        self._df.at[idx, "Route Bucket"] = _safe_str(result.get("route_bucket"))
        self._df.at[idx, "Route Reason"] = _safe_str(result.get("route_reason"))
        self._df.at[idx, "Routed At"] = datetime.now(UTC).strftime("%Y-%m-%d %H:%M")
        self._persist()

    def apply_enrichment(self, email: str, values: dict[str, str]) -> pd.Series | None:
        """Write Clay enrichment columns onto the lead; returns the updated row or None if not found."""
        idx = self._find_email_index(email)
        if idx is None or self._df is None:
            return None
        for col, val in values.items():
            if col not in self._df.columns:
                self._df[col] = ""
            self._df.at[idx, col] = val
        self._persist()
        return self._df.iloc[idx]

    def iter_unrouted_leads(self, limit: int = 50):
        if not self.loaded or self._df is None:
            return
        self._ensure_routing_columns()
        dashboard = self._dashboard_df()
        dashboard = dashboard[~dashboard.apply(is_synthetic_test_lead, axis=1)]
        if "Routed At" in dashboard.columns:
            dashboard = dashboard[dashboard["Routed At"].astype(str).str.strip() == ""]
        dashboard = _sort_dashboard_newest(dashboard)
        for _, row in dashboard.head(limit).iterrows():
            yield row

    def _load_qualified_file(self, path: Path) -> bool:
        try:
            df = pd.read_excel(path, engine="openpyxl")
        except Exception:
            return False
        if "AI Tier" not in df.columns:
            return False
        self.save(df, source=str(path.name), note="Loaded pre-scored qualified export")
        return True

    def save_baseline(self, df: pd.DataFrame, source: str = "baseline.xlsx") -> None:
        """Store previous export for tier comparison and reference scoring anchors."""
        if "AI Tier" not in df.columns:
            raise ValueError("Baseline file must include AI Tier column.")
        self._baseline_df = df.copy()
        self._baseline_meta = {
            "source": source,
            "saved_at": datetime.now(UTC).isoformat(),
            "row_count": len(df),
        }
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        self._baseline_df.to_parquet(BASELINE_PARQUET, index=False)
        self._write_baseline_meta()

        from .reference_scores import save_reference

        save_reference(df)

    def get_compare_summary(self) -> dict[str, Any]:
        from .tier_compare import build_tier_comparison, summarize_comparison

        if not self.loaded or not self.baseline_loaded:
            return {"loaded": False, "baseline_loaded": self.baseline_loaded, "cache_loaded": self.loaded}

        compared = build_tier_comparison(self._baseline_df, self._df)  # type: ignore[arg-type]
        summary = summarize_comparison(compared)
        return {
            "loaded": True,
            "baseline_loaded": True,
            "baseline_source": self._baseline_meta.get("source"),
            **summary,
        }

    def export_compare_dataframe(self, tier: str | None = None) -> pd.DataFrame:
        from .tier_compare import REVIEW_COLUMNS, build_tier_comparison

        if not self.loaded:
            raise ValueError("No scored leads in cache.")
        if not self.baseline_loaded:
            raise ValueError("No baseline export loaded. Upload previous qualified.xlsx in Settings.")

        compared = build_tier_comparison(self._baseline_df, self._df)  # type: ignore[arg-type]
        if tier and tier.lower() != "all":
            compared = compared[compared["AI Tier"].astype(str).str.strip() == tier]

        front = [
            "Email",
            "First Name",
            "Last Name",
            "Lifecycle Stage",
            "Previous AI Tier",
            "AI Tier",
            "Tier Change",
            "Previous AI Score",
            "AI Score",
            "Score Delta",
            "Hot Change Bucket",
            "Change Tag",
            "AI Reasons",
        ] + REVIEW_COLUMNS
        cols = [c for c in front if c in compared.columns] + [c for c in compared.columns if c not in front]
        return compared[cols].copy()

    def _dashboard_df(self) -> pd.DataFrame:
        """Leads shown on dashboard — excludes Customers (training data only)."""
        if not self.loaded:
            return pd.DataFrame()
        assert self._df is not None
        if "Lifecycle Stage" not in self._df.columns:
            return self._df.copy()
        mask = self._df["Lifecycle Stage"].astype(str).str.strip() != "Customer"
        return self._df[mask].copy()

    def _customer_count(self) -> int:
        if not self.loaded or self._df is None or "Lifecycle Stage" not in self._df.columns:
            return 0
        return int((self._df["Lifecycle Stage"].astype(str).str.strip() == "Customer").sum())

    def save(self, df: pd.DataFrame, source: str = "upload", note: str = "") -> None:
        self._df = df.copy()
        self._meta = {
            "source": source,
            "note": note,
            "scored_at": datetime.now(UTC).isoformat(),
            "row_count": len(df),
            "customers_excluded_from_dashboard": self._customer_count(),
        }
        self._persist()

    def _find_email_index(self, email: str) -> int | None:
        if not self.loaded or not email:
            return None
        assert self._df is not None
        if "Email" not in self._df.columns:
            return None
        normalized = email.strip().lower()
        matches = self._df[
            self._df["Email"].astype(str).str.strip().str.lower() == normalized
        ]
        if matches.empty:
            return None
        return int(matches.index[0])

    def append_scored_row(self, row: pd.Series | dict[str, Any]) -> dict[str, Any]:
        """Append or update a pre-scored lead in cache."""
        if self._df is None:
            self._df = pd.DataFrame()

        series = pd.Series(row) if isinstance(row, dict) else row
        email = _safe_str(series.get("Email", ""))

        if email and len(self._df) > 0 and "Email" in self._df.columns:
            existing_idx = self._find_email_index(email)
            if existing_idx is not None:
                for col, val in series.items():
                    self._df.at[existing_idx, col] = val
                action = "updated"
                row_idx = existing_idx
            else:
                self._df = pd.concat([self._df, series.to_frame().T], ignore_index=True)
                action = "created"
                row_idx = len(self._df) - 1
        else:
            self._df = pd.concat([self._df, series.to_frame().T], ignore_index=True)
            action = "created"
            row_idx = len(self._df) - 1

        self._meta["scored_at"] = datetime.now(UTC).isoformat()
        self._meta["row_count"] = len(self._df)
        self._meta["customers_excluded_from_dashboard"] = self._customer_count()
        self._meta["last_append"] = datetime.now(UTC).isoformat()
        self._persist()

        result_row = self._df.iloc[row_idx]
        return {
            "action": action,
            "row_index": int(row_idx),
            "ai_score": float(result_row.get("AI Score", 0)),
            "ai_tier": _safe_str(result_row.get("AI Tier", "")),
            "email": email,
            "on_dashboard": not _is_customer(result_row),
        }

    async def append_lead(
        self,
        row: dict[str, Any],
        use_llm: bool = True,
        *,
        form_config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Score a single lead and append to cache."""
        from .routing import _form_auto_route
        from .scorer import score_dataframe_async

        df = pd.DataFrame([row])
        scored = await score_dataframe_async(df, use_llm=use_llm)
        if "Source" not in scored.columns or not _safe_str(scored.iloc[0].get("Source", "")):
            scored.at[scored.index[0], "Source"] = row.get("Source", "Wufoo")
        if "Create Date" not in scored.columns or not _safe_str(scored.iloc[0].get("Create Date", "")):
            scored.at[scored.index[0], "Create Date"] = datetime.now(UTC).strftime("%Y-%m-%d %H:%M")
        # Keep all Wufoo form columns on the scored row (email, routing, HubSpot/n8n).
        for col, val in row.items():
            if val is None or not str(val).strip():
                continue
            if col not in scored.columns:
                scored[col] = ""
            scored.at[scored.index[0], col] = val

        result = self.append_scored_row(scored.iloc[0])
        log_webhook_lead(
            record_id=_safe_str(scored.iloc[0].get("Record ID", "")),
            email=result.get("email", ""),
        )

        from .routing import route_and_notify
        from .routing_config import load_routing_config

        config = load_routing_config()
        if _form_auto_route(form_config, config) and result.get("email"):
            row_series = self._df.iloc[result["row_index"]]
            route_result = route_and_notify(row_series, config, form_config=form_config)
            if route_result.get("assigned"):
                self.apply_routing_result(result["email"], route_result)
            result["routing"] = route_result

        return result

    def reapply_tier_labels(self) -> int:
        """Recompute AI Tier and Recommended Action from existing AI Score using current rubric."""
        if not self.loaded or self._df is None or self._df.empty:
            return 0

        from .scorer import recommended_action, score_to_tier

        updated = 0
        for idx in self._df.index:
            try:
                score = float(self._df.at[idx, "AI Score"])
            except (TypeError, ValueError):
                continue

            lifecycle = _safe_str(self._df.at[idx, "Lifecycle Stage"])
            if lifecycle == "Customer":
                tier = "Hot"
            else:
                tier = score_to_tier(score)

            action = recommended_action(tier, lifecycle)
            if _safe_str(self._df.at[idx, "AI Tier"]) != tier:
                updated += 1
            self._df.at[idx, "AI Tier"] = tier
            self._df.at[idx, "Recommended Action"] = action

        self._meta["scored_at"] = datetime.now(UTC).isoformat()
        self._persist()
        return updated

    def get_stats(self) -> dict[str, Any]:
        if not self.loaded:
            return {
                "loaded": False,
                "total_leads": 0,
                "tier_counts": {},
                "average_score": 0,
                "customers_excluded": 0,
                "meta": self._meta,
            }

        dashboard = self._dashboard_df()
        summary = metrics_summary(dashboard)
        tier_stats: dict[str, Any] = {}
        for tier in ["Hot", "Warm", "Cold", "Unqualified"]:
            subset = dashboard[dashboard["AI Tier"].astype(str).str.strip() == tier]
            if subset.empty:
                continue
            tier_stats[tier] = {
                "count": int(len(subset)),
                "avg_ai_score": round(float(subset["AI Score"].mean()), 1),
                "avg_ml_score": round(float(subset["ML Score"].mean()), 1)
                if "ML Score" in subset
                else None,
                "avg_llm_score": round(float(subset["LLM Score"].mean()), 1)
                if "LLM Score" in subset
                else None,
            }

        return {
            "loaded": True,
            "total_leads": summary["total_leads"],
            "tier_counts": summary["tier_counts"],
            "average_score": summary["average_score"],
            "tier_stats": tier_stats,
            "customers_excluded": self._customer_count(),
            "meta": self._meta,
        }

    def get_leads(
        self,
        tier: str | None = None,
        page: int = 1,
        limit: int = 50,
        search: str | None = None,
    ) -> dict[str, Any]:
        if not self.loaded:
            return {"total": 0, "page": page, "limit": limit, "leads": []}

        filtered = self._dashboard_df()

        if tier and tier.lower() != "all":
            filtered = filtered[filtered["AI Tier"].astype(str).str.strip() == tier]

        if search:
            q = search.lower().strip()
            mask = pd.Series(False, index=filtered.index)
            for col in ["First Name", "Last Name", "Email", "AI Reasons", "Job Title"]:
                if col in filtered.columns:
                    mask |= filtered[col].astype(str).str.lower().str.contains(q, na=False)
            filtered = filtered[mask]

        filtered = _sort_dashboard_newest(filtered)
        total = len(filtered)
        start = max(0, (page - 1) * limit)
        page_df = filtered.iloc[start : start + limit]

        cols = [c for c in DISPLAY_COLUMNS if c in page_df.columns]
        leads = page_df[cols].fillna("").to_dict(orient="records")

        return {"total": total, "page": page, "limit": limit, "leads": leads}

    def _lookup_dashboard_lead(self, record_id: str = "", email: str = "") -> pd.Series | None:
        dashboard = self._dashboard_df()
        if dashboard.empty:
            return None

        record_key = _safe_str(record_id)
        if record_key and "Record ID" in dashboard.columns:
            matches = dashboard[dashboard["Record ID"].astype(str).str.strip() == record_key]
            if not matches.empty:
                row = matches.iloc[0]
                if not is_synthetic_test_lead(row):
                    return row

        email_key = _norm_email(email)
        if email_key and "Email" in dashboard.columns:
            matches = dashboard[
                dashboard["Email"].astype(str).str.strip().str.lower() == email_key
            ]
            if not matches.empty:
                row = matches.iloc[0]
                if not is_synthetic_test_lead(row):
                    return row

        return None

    def _collect_webhook_lead_rows(self, limit: int | None = None) -> list[pd.Series]:
        if not self.loaded:
            return []

        entries = _read_webhook_log()
        if not entries:
            return []

        seen: set[str] = set()
        rows: list[pd.Series] = []
        for entry in entries:
            if limit is not None and len(rows) >= limit:
                break
            record_id = _safe_str(entry.get("record_id", ""))
            email = _norm_email(entry.get("email", ""))
            dedupe_key = email or record_id
            if not dedupe_key or dedupe_key in seen:
                continue

            row = self._lookup_dashboard_lead(record_id=record_id, email=email)
            if row is None:
                continue

            seen.add(dedupe_key)
            rows.append(row)

        return rows

    def get_recent_webhook_tier_counts(self) -> dict[str, int]:
        """Count webhook-ingested leads by tier (for dashboard incoming badges)."""
        rows = self._collect_webhook_lead_rows(limit=None)
        counts = {"Hot": 0, "Warm": 0, "Cold": 0, "Unqualified": 0}
        for row in rows:
            tier = _safe_str(row.get("AI Tier", ""))
            if tier in counts:
                counts[tier] += 1
        return {"total": len(rows), **counts}

    def get_recent(self, limit: int = 10) -> list[dict[str, Any]]:
        """Recent leads from Wufoo webhooks only — does not include bulk imports."""
        rows = self._collect_webhook_lead_rows(limit=limit)
        if not rows:
            return []

        recent = pd.DataFrame(rows)
        cols = [c for c in DISPLAY_COLUMNS if c in recent.columns]
        return recent[cols].fillna("").to_dict(orient="records")

    def get_all_scored_df(self) -> pd.DataFrame:
        if not self.loaded or self._df is None:
            raise ValueError("No scored leads in cache.")
        return self._df.copy()

    def export_dataframe(self, tier: str | None = None) -> pd.DataFrame:
        if not self.loaded:
            raise ValueError("No scored leads in cache.")

        export_df = self._dashboard_df()
        if tier and tier.lower() != "all":
            export_df = export_df[export_df["AI Tier"].astype(str).str.strip() == tier]
        return export_df.copy()


store = ScoredLeadsStore()
