"""Load and persist editable sales routing rules."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from .config import BASE_DIR, DATA_DIR

DEFAULT_RULES_PATH = BASE_DIR / "config" / "routing_rules.json"
ROUTING_CONFIG_PATH = DATA_DIR / "routing_config.json"


def _safe_str(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def default_rules() -> dict[str, Any]:
    if DEFAULT_RULES_PATH.exists():
        return json.loads(DEFAULT_RULES_PATH.read_text(encoding="utf-8"))
    return {
        "routing_mode": "hybrid",
        "distribution_gene_pct": 10,
        "distribution_jake_pct": 20,
        "distribution_general_pct": 50,
        "distribution_automation_pct": 20,
        "min_leads_for_percentile": 15,
        "automation_skip_notify": True,
        "auto_route_enabled": True,
        "send_email_on_route": True,
        "urgent_min_score": 75,
        "jake_min_warm_score": 70,
        "jake_max_consecutive": 2,
        "reps": [],
    }


def _strip_legacy_routing_keys(config: dict[str, Any]) -> dict[str, Any]:
    config.pop("west_coast_states", None)
    for rep in config.get("reps", []):
        if isinstance(rep, dict):
            rep.pop("west_coast_priority", None)
    return config


def _default_rep_by_id() -> dict[str, dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for rep in default_rules().get("reps", []):
        if isinstance(rep, dict):
            rid = _safe_str(rep.get("id"))
            if rid:
                by_id[rid] = rep
    return by_id


def _backfill_rep_fields(config: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    """Copy hubspot_owner_id and phone from config/routing_rules.json when missing on a rep."""
    defaults = _default_rep_by_id()
    changed = False
    for rep in config.get("reps", []):
        if not isinstance(rep, dict):
            continue
        rid = _safe_str(rep.get("id"))
        if not rid:
            continue
        fallback = defaults.get(rid, {})
        if not _safe_str(rep.get("hubspot_owner_id")):
            owner = _safe_str(fallback.get("hubspot_owner_id"))
            if owner:
                rep["hubspot_owner_id"] = owner
                changed = True
        if not _safe_str(rep.get("phone")):
            phone = _safe_str(fallback.get("phone"))
            if phone:
                rep["phone"] = phone
                changed = True
        if rep.get("weight") in (None, "") and fallback.get("weight") is not None:
            rep["weight"] = fallback.get("weight")
            changed = True
    return config, changed


def load_routing_config() -> dict[str, Any]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not ROUTING_CONFIG_PATH.exists():
        if DEFAULT_RULES_PATH.exists():
            shutil.copy2(DEFAULT_RULES_PATH, ROUTING_CONFIG_PATH)
        else:
            ROUTING_CONFIG_PATH.write_text(
                json.dumps(default_rules(), indent=2), encoding="utf-8"
            )
    config = json.loads(ROUTING_CONFIG_PATH.read_text(encoding="utf-8"))
    config = _strip_legacy_routing_keys(config)
    config, changed = _backfill_rep_fields(config)
    if changed:
        ROUTING_CONFIG_PATH.write_text(json.dumps(config, indent=2), encoding="utf-8")
    return config


def save_routing_config(config: dict[str, Any]) -> dict[str, Any]:
    if "reps" not in config or not isinstance(config["reps"], list):
        raise ValueError("Config must include a reps array.")
    for rep in config["reps"]:
        if _safe_str(rep.get("bucket")) == "automation":
            continue
        if not _safe_str(rep.get("email")):
            raise ValueError(f"Rep {rep.get('name', '?')} must have an email.")
        if not _safe_str(rep.get("id")):
            raise ValueError(f"Rep {rep.get('name', '?')} must have an id.")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    cleaned = _strip_legacy_routing_keys(dict(config))
    ROUTING_CONFIG_PATH.write_text(json.dumps(cleaned, indent=2), encoding="utf-8")
    return cleaned


def get_rep_by_id(config: dict[str, Any], rep_id: str) -> dict[str, Any] | None:
    for rep in config.get("reps", []):
        if _safe_str(rep.get("id")) == rep_id:
            return rep
    return None


def reps_for_bucket(config: dict[str, Any], bucket: str) -> list[dict[str, Any]]:
    return [r for r in config.get("reps", []) if _safe_str(r.get("bucket")) == bucket]
