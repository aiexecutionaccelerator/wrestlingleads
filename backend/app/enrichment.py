"""Lead enrichment with Claude + web search: wrestler record, club, and a pre-call brief for the rep.

Runs synchronously inside route_and_notify (bounded by ENRICHMENT_TIMEOUT_SECONDS) so the brief is
included in the single assignment email/SMS/n8n payload. Enabled when ANTHROPIC_API_KEY is set
(disable with ENRICHMENT_PROVIDER=off).
"""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import UTC, datetime
from typing import Any

import pandas as pd

from .features import _safe_str

logger = logging.getLogger(__name__)

ENRICHMENT_MODEL = os.getenv("ENRICHMENT_MODEL", "claude-opus-5")
MAX_SEARCHES = int(os.getenv("ENRICHMENT_MAX_SEARCHES", "8"))
MAX_FETCHES = int(os.getenv("ENRICHMENT_MAX_FETCHES", "8"))
TIMEOUT_SECONDS = float(os.getenv("ENRICHMENT_TIMEOUT_SECONDS", "150"))

ENRICHMENT_FIELDS: tuple[tuple[str, str], ...] = (
    # (result key, cache column)
    ("wrestler_name", "Wrestler Name"),
    ("parent_name", "Parent Name"),
    ("tw_record", "TW Record"),
    ("tw_weight_class", "TW Weight Class"),
    ("tw_team", "TW Team"),
    ("tw_source_url", "TW Source URL"),
    ("club", "Club Affiliation"),
    ("parent_linkedin_url", "Parent LinkedIn"),
    ("parent_headline", "Parent Headline"),
    ("confidence", "Enrichment Confidence"),
    ("summary", "Enrichment Summary"),
)
RESULT_KEYS = tuple(key for key, _ in ENRICHMENT_FIELDS)
ENRICHMENT_COLUMNS: tuple[str, ...] = tuple(col for _, col in ENRICHMENT_FIELDS) + ("Enriched At",)

SYSTEM_PROMPT = """You research inbound leads for Wrestling Mindset, a 1-on-1 mental-performance coaching program for youth and high-school wrestlers, so the assigned sales rep is prepared before their first call.

You receive one lead's form submission. Do the following, using web search and page fetches as needed:

1. Determine who is who. The form's Name field is sometimes the parent and sometimes the wrestler.
   - Buyer type "Wrestler Seeking..." → the name is the wrestler.
   - Buyer type "Parent..." → the name is the parent, unless the message clearly uses that same first name for the athlete (e.g. Name "Jessie Murphy" + message "Jessie wrestled for one year..." → Jessie is the wrestler). Also extract any child's name given in the message, and use the email address as a clue to the parent's name (e.g. "ronandpamfollett@..." → parents are likely Ron and Pam Follett — report as "possibly Ron and Pam Follett"). If the wrestler's name cannot be determined, leave wrestler_name empty — do not guess.
2. If you have a wrestler name: search TrackWrestling.com (fallback: FloWrestling, school athletics pages) for that wrestler in the given state and grade level. Report the most recent season's win-loss record, weight class, team/school, and the source URL. Only report a match if name AND state agree — never guess between same-name wrestlers. Youth and middle-school coverage is thin; "not found" is a normal outcome.
3. If you have a wrestler name: identify their club/team via USA Wrestling club listings, TrackWrestling team pages, or club rosters. Report a club only if the wrestler's name appears on that club's roster or results.
4. Write a 3-line pre-call brief for the rep, plain text, no markdown. Each line is ONE short sentence (max ~30 words):
   Line 1: wrestler name (or "wrestler name not on form — ask on call"), grade, experience, record/weight/team if found, otherwise "no TrackWrestling record found".
   Line 2: club if found; otherwise the parent's own words from the form message.
   Line 3: reason for inquiry and goal, plus the single most useful opening question or fact for the call.
   Be factual. Mark low-confidence facts as "possibly". Never state something as fact that you did not verify on a page.

Budget: use at most a handful of searches. Stop as soon as you have an answer or it is clear nothing reliable exists.

Return ONLY a JSON object with exactly these keys (use "" for unknown):
wrestler_name, parent_name, tw_record, tw_weight_class, tw_team, tw_source_url, club, parent_linkedin_url, parent_headline, confidence, summary
- confidence: one of "High", "Medium", "Low", "Not found" (High = record verified on TrackWrestling with matching state; Medium = record or club found with minor ambiguity; Low = only weak signals; Not found = nothing verified).
- summary: the 3-line brief, lines separated by "\\n".
- parent_linkedin_url / parent_headline: only if you happened to find a clearly matching public profile; otherwise "".
"""


# ---------------------------------------------------------------- config


def enrichment_secret() -> str:
    return os.getenv("ENRICHMENT_SECRET", "").strip() or os.getenv("CLAY_CALLBACK_SECRET", "").strip()


def enrichment_enabled() -> bool:
    if os.getenv("ENRICHMENT_PROVIDER", "").strip().lower() == "off":
        return False
    return bool(os.getenv("ANTHROPIC_API_KEY", "").strip())


# ---------------------------------------------------------------- Claude research


def _lead_description(row: pd.Series | dict[str, Any], rep: dict[str, Any] | None) -> str:
    get = row.get if isinstance(row, dict) else row.get
    first = _safe_str(get("First Name", ""))
    last = _safe_str(get("Last Name", ""))
    fields = [
        ("Name on form", f"{first} {last}".strip()),
        ("Email on form", _safe_str(get("Email", ""))),
        ("Phone on form", _safe_str(get("Phone Number", ""))),
        ("Buyer type", _safe_str(get("Job Title", ""))),
        ("State", _safe_str(get("State/Region", ""))),
        ("Wrestler's grade", _safe_str(get("Wrestler's Grade", ""))),
        ("Years of experience", _safe_str(get("Years experience", ""))),
        ("Wrestler's goal", _safe_str(get("Wrestler's Goal", ""))),
        ("Reason for inquiry", _safe_str(get("Job function", ""))),
        ("Deadline", _safe_str(get("Deadline for Goal", ""))),
        ("Investment level", _safe_str(get("Investment Level", ""))),
        ("Message", _safe_str(get("Message", ""))),
        ("Assigned rep", _safe_str((rep or {}).get("name", ""))),
    ]
    lines = [f"{label}: {value}" for label, value in fields if value]
    return "Lead form submission:\n" + "\n".join(lines)


def _extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object in enrichment response: {text[:200]}")
    return json.loads(match.group(0))


def _tools() -> list[dict[str, Any]]:
    return [
        {"type": "web_search_20260209", "name": "web_search", "max_uses": MAX_SEARCHES},
        {"type": "web_fetch_20260209", "name": "web_fetch", "max_uses": MAX_FETCHES},
    ]


def research_lead_with_claude(
    row: pd.Series | dict[str, Any],
    rep: dict[str, Any] | None = None,
    *,
    client: Any | None = None,
) -> dict[str, str]:
    """One Claude call with web search/fetch; returns a dict keyed by RESULT_KEYS."""
    if client is None:
        import anthropic

        client = anthropic.Anthropic(timeout=TIMEOUT_SECONDS, max_retries=1)

    messages: list[dict[str, Any]] = [{"role": "user", "content": _lead_description(row, rep)}]
    response = None
    for _ in range(4):  # pause_turn continuations
        response = client.messages.create(
            model=ENRICHMENT_MODEL,
            max_tokens=8000,
            system=SYSTEM_PROMPT,
            tools=_tools(),
            messages=messages,
        )
        if response.stop_reason != "pause_turn":
            break
        messages = [messages[0], {"role": "assistant", "content": response.content}]

    if response is None:
        raise RuntimeError("Enrichment produced no response.")
    if response.stop_reason == "refusal":
        raise RuntimeError("Enrichment request was declined by the model.")

    text = "".join(block.text for block in response.content if getattr(block, "type", "") == "text")
    parsed = _extract_json(text)
    result = {key: _safe_str(parsed.get(key, "")) for key in RESULT_KEYS}

    usage = getattr(response, "usage", None)
    server_use = getattr(usage, "server_tool_use", None)
    get = row.get if isinstance(row, dict) else row.get
    logger.info(
        "Enrichment done email=%s confidence=%s searches=%s in=%s out=%s",
        _safe_str(get("Email", "")),
        result.get("confidence"),
        getattr(server_use, "web_search_requests", None),
        getattr(usage, "input_tokens", None),
        getattr(usage, "output_tokens", None),
    )
    return result


# ---------------------------------------------------------------- store + notify


def result_to_columns(result: dict[str, Any]) -> dict[str, str]:
    values = {col: _safe_str(result.get(key, "")) for key, col in ENRICHMENT_FIELDS}
    values["Enriched At"] = datetime.now(UTC).strftime("%Y-%m-%d %H:%M")
    return values


def record_enrichment(email: str, result: dict[str, Any]) -> dict[str, Any]:
    """Store enrichment values on the lead in the cache."""
    from .store import store

    values = result_to_columns(result)
    row = store.apply_enrichment(email, values)
    if row is None:
        return {"stored": False, "reason": f"No lead in cache with email {email}."}
    return {"stored": True, "stored_fields": [k for k, v in values.items() if v]}


def enrich_lead(row: pd.Series | dict[str, Any], rep: dict[str, Any] | None) -> dict[str, Any]:
    """Research + store; returns {"result", "stored", ...}. Raises on research failure."""
    get = row.get if isinstance(row, dict) else row.get
    email = _safe_str(get("Email", ""))
    result = research_lead_with_claude(row, rep)
    outcome = record_enrichment(email, result)
    return {"result": result, **outcome}


def enrich_before_notify(row: pd.Series | dict[str, Any], rep: dict[str, Any] | None) -> dict[str, str] | None:
    """Best-effort enrichment for the assignment notification.

    Returns the enrichment columns to merge into the row used for email/SMS/n8n, or None if
    research failed or timed out (the notification then goes out without a brief).
    """
    get = row.get if isinstance(row, dict) else row.get
    email = _safe_str(get("Email", ""))
    try:
        result = research_lead_with_claude(row, rep)
    except Exception:
        logger.exception("Enrichment failed for email=%s — notifying without brief", email)
        return None
    values = result_to_columns(result)
    try:
        record_enrichment(email, result)
    except Exception:
        logger.exception("Enrichment could not be stored for email=%s", email)
    return values
