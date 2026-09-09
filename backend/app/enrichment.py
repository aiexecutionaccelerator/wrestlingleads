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
import time
from datetime import UTC, datetime
from typing import Any

import httpx
import pandas as pd

from .features import _safe_str

logger = logging.getLogger(__name__)

ENRICHMENT_MODEL = os.getenv("ENRICHMENT_MODEL", "claude-opus-5")
MAX_SEARCHES = int(os.getenv("ENRICHMENT_MAX_SEARCHES", "5"))
MAX_FETCHES = int(os.getenv("ENRICHMENT_MAX_FETCHES", "4"))
FETCH_MAX_TOKENS = int(os.getenv("ENRICHMENT_FETCH_MAX_TOKENS", "15000"))
TIMEOUT_SECONDS = float(os.getenv("ENRICHMENT_TIMEOUT_SECONDS", "150"))

ENRICHMENT_FIELDS: tuple[tuple[str, str], ...] = (
    # (result key, cache column)
    ("wrestler_name", "Wrestler Name"),
    ("parent_name", "Parent Name"),
    ("tw_record", "TW Record"),
    ("tw_weight_class", "TW Weight Class"),
    ("tw_team", "TW Team"),
    ("tw_source_url", "TW Source URL"),
    ("flo_profile_url", "Flo Profile"),
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
2. If you have a wrestler name: FIRST call search_flo_athletes (FloWrestling's athlete database — fast and structured). Pick the candidate whose hometown state matches the lead's state, then call get_flo_athlete for their record. Verify plausibility before matching: the athlete's level and most recent season must fit the lead's grade and experience (a college wrestler with seasons from years ago is NOT a current middle/high schooler with the same name — reject the match). If Flo has nothing plausible, fall back to web search of TrackWrestling.com and school athletics pages. Only report a match if name AND state agree and the level/era is plausible — never guess between same-name wrestlers. Youth coverage is thin; "not found" is a normal outcome. Use the Flo profile URL (or TrackWrestling page) as tw_source_url; tw_record/tw_weight_class/tw_team may come from either source.
3. If you have a wrestler name: identify their club. The best evidence is get_flo_athlete's teams_in_results — at club and national tournaments the bout "team" IS the wrestler's club (e.g. "Keystone Wrestling Academy"), while school duals list the school; report the non-school team there as the club. Wrestlers often have DUPLICATE Flo profiles — run get_flo_athlete on EVERY search candidate matching the lead's state and merge what you learn (one profile may hold school results, another club results). If Flo shows nothing, fall back to USA Wrestling club listings, TrackWrestling team pages, or club rosters. Report a club only if the wrestler's name appears in that club's results or roster.
4. Write a 3-line pre-call brief for the rep, plain text, no markdown. Each line is ONE short sentence (max ~30 words):
   Line 1: wrestler name (or "wrestler name not on form — ask on call"), grade, experience, record/weight/team if found, otherwise "no TrackWrestling record found".
   Line 2: club if found; otherwise the parent's own words from the form message.
   Line 3: reason for inquiry and goal, plus the single most useful opening question or fact for the call.
   Be factual. Mark low-confidence facts as "possibly". Never state something as fact that you did not verify on a page.

Budget: use at most a handful of searches. Stop as soon as you have an answer or it is clear nothing reliable exists.

Return ONLY a JSON object with exactly these keys (use "" for unknown):
wrestler_name, parent_name, tw_record, tw_weight_class, tw_team, tw_source_url, flo_profile_url, club, parent_linkedin_url, parent_headline, confidence, summary
- flo_profile_url: the FloWrestling profile URL of the matched wrestler (from search_flo_athletes / get_flo_athlete), only if the match is plausible; otherwise "".
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


FLO_API = "https://prod-web-api.flowrestling.org/api"
_FLO_HEADERS = {"User-Agent": "Mozilla/5.0", "Content-Type": "application/json"}


def _flo_search(name: str) -> dict[str, Any]:
    """FloWrestling athlete search -> compact candidate list."""
    response = httpx.post(
        f"{FLO_API}/search",
        headers=_FLO_HEADERS,
        json={"offsetPerGroup": 0, "search": name, "limitPerGroup": 20, "entities": ["athlete"]},
        timeout=15.0,
    )
    response.raise_for_status()
    groups = response.json().get("data") or []
    items = groups[0].get("items", []) if groups else []
    return {
        "candidates": [
            {
                "athlete_id": item.get("id"),
                "name": item.get("title"),
                "team": item.get("metadata1"),
                "hometown": item.get("metadata2"),
                "profile_url": item.get("url"),
            }
            for item in items[:20]
        ]
    }


def _flo_athlete_details(athlete_id: str) -> dict[str, Any]:
    """FloWrestling athlete profile + win-loss stats, trimmed for the model."""
    profile = httpx.get(f"{FLO_API}/athletes/{athlete_id}", headers=_FLO_HEADERS, timeout=15.0)
    profile.raise_for_status()
    p = profile.json().get("data") or {}
    out: dict[str, Any] = {
        "name": f"{p.get('firstName', '')} {p.get('lastName', '')}".strip(),
        "team": (p.get("team") or {}).get("name"),
        "level": (p.get("team") or {}).get("level"),
        "weight_class": p.get("weightClass"),
        "hometown": p.get("hometown"),
        "profile_url": f"https://www.flowrestling.org/people/{athlete_id}",
    }
    try:
        stats = httpx.get(f"{FLO_API}/athletes/{athlete_id}/stats", headers=_FLO_HEADERS, timeout=15.0)
        stats.raise_for_status()
        st = stats.json().get("data") or {}
        seasons = [
            {"season": season.get("season"), "wins": season.get("wins"), "losses": season.get("losses")}
            for level in st.get("perLevelStats") or []
            for season in level.get("perSeasonStats") or []
        ]
        seasons.sort(key=lambda x: _safe_str(x.get("season")), reverse=True)
        out["career_record"] = f"{st.get('overallWins')}-{st.get('overallLosses')}"
        out["last_season_record"] = f"{st.get('lastSeasonWins')}-{st.get('lastSeasonLosses')}"
        out["seasons"] = seasons[:6]
    except Exception as exc:
        out["stats_error"] = str(exc)[:120]
    try:
        results = httpx.get(f"{FLO_API}/athletes/{athlete_id}/results", headers=_FLO_HEADERS, timeout=15.0)
        results.raise_for_status()
        # Clubs usually surface here: at club/national tournaments the bout "team" is the club,
        # while school duals list the school. Collect the distinct teams this athlete wrestled for.
        teams: dict[str, dict[str, Any]] = {}
        for event in results.json().get("data") or []:
            for bout in event.get("boutResults") or []:
                team = _safe_str(((bout.get("athlete") or {}).get("team") or {}).get("name"))
                if team and team not in teams:
                    teams[team] = {"team": team, "example_event": event.get("name"), "date": _safe_str(bout.get("date"))[:10]}
        out["teams_in_results"] = list(teams.values())[:6]
    except Exception as exc:
        out["results_error"] = str(exc)[:120]
    return out


def _run_flo_tool(name: str, tool_input: dict[str, Any]) -> dict[str, Any]:
    try:
        if name == "search_flo_athletes":
            return _flo_search(_safe_str(tool_input.get("name")))
        if name == "get_flo_athlete":
            return _flo_athlete_details(_safe_str(tool_input.get("athlete_id")))
        return {"error": f"Unknown tool {name}"}
    except Exception as exc:
        return {"error": str(exc)[:200]}


def _tools() -> list[dict[str, Any]]:
    return [
        {"type": "web_search_20260209", "name": "web_search", "max_uses": MAX_SEARCHES},
        {
            "type": "web_fetch_20260209",
            "name": "web_fetch",
            "max_uses": MAX_FETCHES,
            # Cap ingested page size — uncapped pages were the main token cost per lead.
            "max_content_tokens": FETCH_MAX_TOKENS,
        },
        {
            "name": "search_flo_athletes",
            "description": "Search FloWrestling's athlete database by name. Returns candidates with team, hometown (city, state), and profile URL. Use this FIRST to find the wrestler; disambiguate by state.",
            "input_schema": {
                "type": "object",
                "properties": {"name": {"type": "string", "description": "Wrestler full name, e.g. 'Michael Horton'"}},
                "required": ["name"],
            },
        },
        {
            "name": "get_flo_athlete",
            "description": "Get a FloWrestling athlete's profile and win-loss records (career, last season, per-season) by athlete_id from search_flo_athletes. Check the level and season years are plausible for the lead before treating it as a match.",
            "input_schema": {
                "type": "object",
                "properties": {"athlete_id": {"type": "string"}},
                "required": ["athlete_id"],
            },
        },
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
    started = time.monotonic()
    container_id: str | None = None
    totals = {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0}
    for _ in range(12):  # custom tool calls + pause_turn continuations
        if time.monotonic() - started > TIMEOUT_SECONDS:
            raise TimeoutError(f"enrichment exceeded {TIMEOUT_SECONDS:.0f}s")
        extra: dict[str, Any] = {"container": container_id} if container_id else {}
        response = client.messages.create(
            model=ENRICHMENT_MODEL,
            max_tokens=8000,
            system=SYSTEM_PROMPT,
            tools=_tools(),
            messages=messages,
            # Auto-cache the growing prefix so each loop round re-reads instead of re-billing it.
            extra_body={"cache_control": {"type": "ephemeral"}},
            **extra,
        )
        usage = getattr(response, "usage", None)
        totals["input"] += getattr(usage, "input_tokens", 0) or 0
        totals["output"] += getattr(usage, "output_tokens", 0) or 0
        totals["cache_read"] += getattr(usage, "cache_read_input_tokens", 0) or 0
        totals["cache_write"] += getattr(usage, "cache_creation_input_tokens", 0) or 0
        # The 20260209 web tools run in a server-side container; continuations of a
        # turn that used them must carry the container id or the API returns a 400.
        container = getattr(response, "container", None)
        if container is not None and getattr(container, "id", None):
            container_id = container.id
        if response.stop_reason == "pause_turn":
            messages = messages + [{"role": "assistant", "content": response.content}]
            continue
        if response.stop_reason == "tool_use":
            tool_results = []
            for block in response.content:
                if getattr(block, "type", "") == "tool_use":
                    output = _run_flo_tool(block.name, dict(block.input or {}))
                    tool_results.append(
                        {"type": "tool_result", "tool_use_id": block.id, "content": json.dumps(output)[:12000]}
                    )
            messages = messages + [
                {"role": "assistant", "content": response.content},
                {"role": "user", "content": tool_results},
            ]
            continue
        break

    if response is None:
        raise RuntimeError("Enrichment produced no response.")
    if response.stop_reason == "refusal":
        raise RuntimeError("Enrichment request was declined by the model.")

    text = "".join(block.text for block in response.content if getattr(block, "type", "") == "text")
    parsed = _extract_json(text)
    result = {key: _safe_str(parsed.get(key, "")) for key in RESULT_KEYS}

    # Rough list-price estimate so cost per lead is visible in the logs.
    rates = {"claude-opus-5": (5.0, 25.0), "claude-sonnet-5": (3.0, 15.0)}
    in_rate, out_rate = rates.get(ENRICHMENT_MODEL, (5.0, 25.0))
    est_cost = (
        totals["input"] * in_rate
        + totals["cache_write"] * in_rate * 1.25
        + totals["cache_read"] * in_rate * 0.1
        + totals["output"] * out_rate
    ) / 1_000_000
    get = row.get if isinstance(row, dict) else row.get
    logger.info(
        "Enrichment done email=%s confidence=%s tokens in=%s out=%s cache_read=%s cache_write=%s est_cost=$%.2f",
        _safe_str(get("Email", "")),
        result.get("confidence"),
        totals["input"],
        totals["output"],
        totals["cache_read"],
        totals["cache_write"],
        est_cost,
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


def _short_error(exc: Exception) -> str:
    text = f"{type(exc).__name__}: {exc}"
    if "usage limits" in text.lower() or "spend limit" in text.lower():
        return "the Anthropic monthly spend limit has been reached — raise it under Billing limits in console.anthropic.com"
    if "authentication" in text.lower() or "401" in text or "api key" in text.lower():
        return "the Anthropic API key on the server is invalid or missing — fix ANTHROPIC_API_KEY on Railway"
    if "timeout" in text.lower() or "timed out" in text.lower():
        return f"research timed out after {TIMEOUT_SECONDS:.0f}s"
    if "rate" in text.lower() and "limit" in text.lower():
        return "the Anthropic API rate limit was hit — will work again shortly"
    return text[:160]


def enrich_before_notify(row: pd.Series | dict[str, Any], rep: dict[str, Any] | None) -> tuple[dict[str, str], bool]:
    """Best-effort enrichment for the assignment notification.

    Returns (columns to merge into the row used for email/SMS/n8n, ok). On research failure the
    columns carry an explicit error line so the notification says WHY there is no brief, and ok
    is False (nothing is stored on the lead, no HubSpot note is written).
    """
    get = row.get if isinstance(row, dict) else row.get
    email = _safe_str(get("Email", ""))
    try:
        result = research_lead_with_claude(row, rep)
    except Exception as exc:
        logger.exception("Enrichment failed for email=%s — notifying with error note", email)
        return {
            "Enrichment Confidence": "Error",
            "Enrichment Summary": f"Automatic lead research was unavailable: {_short_error(exc)}",
        }, False
    values = result_to_columns(result)
    try:
        record_enrichment(email, result)
    except Exception:
        logger.exception("Enrichment could not be stored for email=%s", email)
    return values, True
