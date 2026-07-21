"""Optional LLM assist for column mapping, lithology synonyms, and QA narratives."""

from __future__ import annotations

import json
import logging
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Literal, Protocol, Sequence

from ai_quality import ColumnMapping, MappingProposal, QualityIssue
from constants import CANONICAL_LITHOLOGY_CODES
from models import CorrelationOverride, Lithology
from render_theme import DEFAULT_CONSULTING_NOTES, strip_cross_section_prefix

logger = logging.getLogger(__name__)

CANONICAL_COLLAR_COLUMNS = ["hole_id", "easting", "northing", "elevation", "total_depth"]
CANONICAL_LITHOLOGY_COLUMNS = [
    "hole_id",
    "from_depth",
    "to_depth",
    "lithology_code",
    "hatch_pattern",
    "unit_order",
]
CANONICAL_WATER_COLUMNS = ["hole_id", "depth", "elevation_masl"]
CANONICAL_SCREEN_COLUMNS = ["hole_id", "from_depth", "to_depth"]
CANONICAL_GRADIENT_COLUMNS = ["hole_id", "direction"]
_SHEET_CANONICAL_COLUMNS: dict[str, list[str]] = {
    "collars": CANONICAL_COLLAR_COLUMNS,
    "water": CANONICAL_WATER_COLUMNS,
    "screens": CANONICAL_SCREEN_COLUMNS,
    "gradients": CANONICAL_GRADIENT_COLUMNS,
    "lithology": CANONICAL_LITHOLOGY_COLUMNS,
}

_PREFERRED_LITHOLOGY = frozenset({"Sand", "Clay", "Topsoil", "Silt", "Gravel", "Sandstone"})
_LITHOLOGY_RULES: tuple[tuple[tuple[str, ...], str, float], ...] = (
    (("topsoil", "top soil"), "Topsoil", 0.9),
    (("sandstone",), "Sandstone", 0.88),
    (("sand",), "Sand", 0.85),
    (("silty clay", "clayey silt"), "Silty Clay", 0.82),
    (("sandy clay",), "Sandy Clay", 0.82),
    (("fat clay", "clay"), "Clay", 0.88),
    (("silt",), "Silt", 0.8),
    (("gravel",), "Gravel", 0.8),
    (("bedrock", "rock"), "Bedrock", 0.75),
    (("organic", "peat"), "Organics", 0.8),
)

_TRANSECT_LABEL_RE = re.compile(
    r"(?:section\s+)?([A-Za-z])\s*[-–—]\s*\1'?|(?:section\s+)([A-Za-z]-[A-Za-z]')",
    flags=re.IGNORECASE,
)

SHEET_ROLE_OPTIONS = (
    "collars",
    "lithology",
    "water",
    "screens",
    "gradients",
    "deviations",
    "unknown",
)

# Prefer consulting-friendly codes first, then full registry.
LITHOLOGY_SUGGESTION_OPTIONS = tuple(
    sorted(
        CANONICAL_LITHOLOGY_CODES,
        key=lambda code: (0 if code in _PREFERRED_LITHOLOGY else 1, code),
    )
)
_LITHOLOGY_OPTION_SET = frozenset(LITHOLOGY_SUGGESTION_OPTIONS)

_BLOCKS_GENERATE = frozenset(
    {
        "duplicate_lithology_no_unit_order",
        "duplicate_unit_order",
        "depth_overlap",
        "below_td",
        "orphan_lithology",
    }
)

_FIX_CATALOG: dict[str, tuple[str, str, str]] = {
    # action_id, summary template, action text
    "duplicate_lithology_no_unit_order": (
        "auto_unit_order",
        "Duplicate lithology codes lack unit_order",
        "Run auto-assign unit_order from depth (1 = shallowest).",
    ),
    "duplicate_unit_order": (
        "manual_unit_order",
        "Conflicting unit_order values in one hole",
        "Edit the Lithology sheet so each unit_order is unique per hole.",
    ),
    "placeholder_elevation": (
        "relative_elevation",
        "All collars use placeholder elevation",
        "Set surveyed RL, or switch Elevation mode to Relative depth below collar.",
    ),
    "depth_overlap": (
        "manual_intervals",
        "Lithology intervals overlap in depth",
        "Correct from_depth/to_depth so intervals do not overlap.",
    ),
    "below_td": (
        "manual_intervals",
        "Interval extends below total depth",
        "Shorten the interval or increase total_depth on the collar.",
    ),
    "orphan_lithology": (
        "manual_hole_id",
        "Lithology references an unknown hole_id",
        "Add a matching collar or fix the hole_id on the lithology row.",
    ),
    "depth_gap": (
        "review_gaps",
        "Missing depth coverage in the log",
        "Fill gaps in the Lithology sheet or accept as incomplete coverage.",
    ),
    "off_transect": (
        "adjust_transect",
        "Borehole is far from the transect",
        "Raise the offset warning threshold, remove the hole, or redraw the transect.",
    ),
    "no_lithology": (
        "add_lithology",
        "Hole has no lithology intervals",
        "Add lithology rows for this hole_id or drop it from the section.",
    ),
    "flat_collar_grid": (
        "check_coordinates",
        "All collars share identical coordinates",
        "Verify easting/northing (or lat/long) in the source workbook.",
    ),
    "single_interval": (
        "info_only",
        "Hole has only one lithology interval",
        "No action required unless more detail is available in field logs.",
    ),
}


@dataclass(frozen=True)
class LithologySuggestion:
    source_code: str
    canonical_code: str
    confidence: float
    rationale: str


@dataclass(frozen=True)
class FixStep:
    issue_code: str
    hole_id: str | None
    summary: str
    blocks_generate: bool
    action: str
    action_id: str


@dataclass(frozen=True)
class ReportMetadataSuggestion:
    section_label: str
    map_scale: str
    notes: tuple[str, ...]
    figure_caption: str
    prepared_for: str = ""
    prepared_by: str = ""
    source: str = ""
    project_number: str = ""
    transect_start_label: str = ""
    transect_end_label: str = ""


@dataclass(frozen=True)
class CorrelationSuggestion:
    left_hole_id: str
    right_hole_id: str
    left_unit_order: int
    right_unit_order: int
    confidence: float
    rationale: str

    def to_override(self) -> CorrelationOverride:
        return CorrelationOverride(
            left_hole_id=self.left_hole_id,
            right_hole_id=self.right_hole_id,
            left_unit_order=self.left_unit_order,
            right_unit_order=self.right_unit_order,
        )


@dataclass(frozen=True)
class SheetRoleSuggestion:
    sheet_name: str
    role: str
    confidence: float
    rationale: str


@dataclass(frozen=True)
class TransectParseResult:
    hole_ids: tuple[str, ...]
    section_label: str
    rationale: str


class LLMProvider(Protocol):
    def complete_json(self, system_prompt: str, user_prompt: str) -> dict:
        ...


LLMProviderKind = Literal["groq", "gemini", "openai"]
DEFAULT_LLM_PROVIDER: LLMProviderKind = "groq"
FREE_LLM_PROVIDERS: tuple[LLMProviderKind, ...] = ("groq", "gemini")
# Prefer free tiers first when auto-detecting from environment keys.
_LLM_PROVIDER_PREFERENCE: tuple[LLMProviderKind, ...] = ("groq", "gemini", "openai")

_LLM_ENV_KEYS: dict[LLMProviderKind, str] = {
    "groq": "GROQ_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "openai": "OPENAI_API_KEY",
}


def _load_json_content(text: str) -> dict:
    """Parse model output, tolerating optional ```json fences."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    payload = json.loads(cleaned or "{}")
    if not isinstance(payload, dict):
        raise ValueError("LLM response JSON must be an object")
    return payload


class OpenAICompatibleProvider:
    """Chat-completions provider for OpenAI and OpenAI-compatible APIs (e.g. Groq)."""

    def __init__(
        self,
        api_key: str,
        *,
        model: str,
        base_url: str | None = None,
        timeout_s: float = 60.0,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self.timeout_s = timeout_s

    def complete_json(self, system_prompt: str, user_prompt: str) -> dict:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("openai package is not installed") from exc

        client_kwargs: dict[str, object] = {"api_key": self.api_key, "timeout": self.timeout_s}
        if self.base_url:
            client_kwargs["base_url"] = self.base_url
        client = OpenAI(**client_kwargs)
        response = client.chat.completions.create(
            model=self.model,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
        )
        content = response.choices[0].message.content or "{}"
        return _load_json_content(content)


class OpenAIProvider(OpenAICompatibleProvider):
    """Paid OpenAI API."""

    def __init__(self, api_key: str, model: str = "gpt-4o-mini") -> None:
        super().__init__(api_key, model=model)


class GroqProvider(OpenAICompatibleProvider):
    """Groq free tier — OpenAI-compatible endpoint."""

    def __init__(self, api_key: str, model: str = "llama-3.1-8b-instant") -> None:
        super().__init__(
            api_key,
            model=model,
            base_url="https://api.groq.com/openai/v1",
        )


class GeminiProvider:
    """Google Gemini free tier via Generative Language API."""

    def __init__(self, api_key: str, model: str = "gemini-2.0-flash") -> None:
        self.api_key = api_key
        self.model = model

    def complete_json(self, system_prompt: str, user_prompt: str) -> dict:
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}"
            ":generateContent"
        )
        body = json.dumps(
            {
                "systemInstruction": {"parts": [{"text": system_prompt}]},
                "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
                "generationConfig": {
                    "temperature": 0.0,
                    "responseMimeType": "application/json",
                },
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=body,
            headers={
                "Content-Type": "application/json",
                "x-goog-api-key": self.api_key,
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            # Never echo API keys that might appear in upstream error payloads.
            safe_detail = detail.replace(self.api_key, "[redacted]")[:500]
            raise RuntimeError(f"Gemini API error {exc.code}: {safe_detail}") from exc
        candidates = payload.get("candidates") or []
        if not candidates:
            raise RuntimeError("Gemini API returned no candidates")
        parts = candidates[0].get("content", {}).get("parts") or []
        if not parts:
            raise RuntimeError("Gemini API returned empty content")
        return _load_json_content(str(parts[0].get("text", "{}")))


def resolve_llm_api_key(provider: LLMProviderKind, api_key: str | None = None) -> str:
    """Return trimmed API key from argument or ``GROQ_API_KEY`` / ``GEMINI_API_KEY`` / ``OPENAI_API_KEY``."""
    key = (api_key or "").strip()
    if key:
        return key
    return os.environ.get(_LLM_ENV_KEYS[provider], "").strip()


def is_free_llm_provider(provider: str | LLMProviderKind) -> bool:
    """True for Groq / Gemini free-tier providers."""
    return str(provider).strip().lower() in FREE_LLM_PROVIDERS


def preferred_llm_provider_from_env() -> LLMProviderKind | None:
    """Return the first provider with an env API key (Groq → Gemini → OpenAI)."""
    for kind in _LLM_PROVIDER_PREFERENCE:
        if resolve_llm_api_key(kind, None):
            return kind
    return None


def build_llm_provider(
    provider: str | LLMProviderKind = DEFAULT_LLM_PROVIDER,
    api_key: str | None = None,
) -> LLMProvider | None:
    """Construct an LLM provider when a key is available; otherwise return None."""
    kind: LLMProviderKind
    normalized = (provider or DEFAULT_LLM_PROVIDER).strip().lower()
    if normalized not in _LLM_ENV_KEYS:
        logger.warning("Unknown LLM provider %r; defaulting to Groq", provider)
        kind = DEFAULT_LLM_PROVIDER
    else:
        kind = normalized  # type: ignore[assignment]
    key = resolve_llm_api_key(kind, api_key)
    if not key:
        return None
    if kind == "gemini":
        return GeminiProvider(key)
    if kind == "openai":
        return OpenAIProvider(key)
    return GroqProvider(key)


_MOCK_DEFAULTS: dict[str, dict] = {
    "fix_plan": {
        "steps": [
            {
                "issue_code": "duplicate_lithology_no_unit_order",
                "hole_id": "BH-01",
                "summary": "Duplicate codes lack unit_order",
                "blocks_generate": True,
                "action": "Run auto-assign unit_order from depth.",
                "action_id": "auto_unit_order",
            }
        ]
    },
    "report": {
        "section_label": "A-A'",
        "map_scale": "1:1000",
        "notes": list(DEFAULT_CONSULTING_NOTES),
        "figure_caption": "Cross section A-A' through monitoring wells.",
        "prepared_for": "",
        "prepared_by": "",
        "source": "",
        "project_number": "",
        "transect_start_label": "A",
        "transect_end_label": "A'",
    },
    "correlation": {
        "suggestions": [
            {
                "left_hole_id": "BH-01",
                "right_hole_id": "BH-02",
                "left_unit_order": 1,
                "right_unit_order": 2,
                "confidence": 0.8,
                "rationale": "Same lithology at similar depth rank",
            }
        ]
    },
    "sheets": {
        "suggestions": [
            {
                "sheet_name": "BH_Collars",
                "role": "collars",
                "confidence": 0.9,
                "rationale": "Contains easting/northing headers",
            }
        ]
    },
    "transect": {
        "hole_ids": ["MW-01", "MW-02"],
        "section_label": "B-B'",
        "rationale": "Ordered holes named in the request",
    },
    "qa": {"answer": "MW-03 has no water measurement (NM)."},
    "columns": {
        "mappings": [{"source": "BH", "canonical": "hole_id", "confidence": 0.95}],
    },
    "lithology": {
        "suggestions": [
            {
                "source": "Fat Clay",
                "canonical": "Clay",
                "confidence": 0.92,
                "rationale": "Common field log descriptor",
            }
        ]
    },
    "narrative": {
        "summary": "Data quality review found depth gaps and one off-transect borehole."
    },
}

_MOCK_ROUTES: tuple[tuple[tuple[str, ...], str], ...] = (
    (("fix plan", "fix steps"), "fix_plan"),
    (("report metadata", "title block"), "report"),
    (("correlation",), "correlation"),
    (("sheet role", "sheet roles"), "sheets"),
    (("column", "header"), "columns"),
    (("lithology",), "lithology"),
)


class MockLLMProvider:
    """Deterministic provider for tests."""

    def __init__(self, responses: dict[str, dict] | None = None) -> None:
        self.responses = responses or {}

    def complete_json(self, system_prompt: str, user_prompt: str) -> dict:
        text = user_prompt.lower()
        if "transect" in text and "hole" in text:
            key = "transect"
        elif "answer" in text or "question" in text:
            key = "qa"
        else:
            key = "narrative"
            for needles, route in _MOCK_ROUTES:
                if any(needle in text for needle in needles):
                    key = route
                    break
        return self.responses.get(key, _MOCK_DEFAULTS[key])


class AIAssistant:
    """Hybrid assistant: local rules first, LLM suggestions when enabled."""

    def __init__(self, provider: LLMProvider | None = None) -> None:
        self.provider = provider

    @property
    def enabled(self) -> bool:
        return self.provider is not None

    def _complete_json(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        label: str,
    ) -> dict | None:
        if self.provider is None:
            return None
        try:
            return self.provider.complete_json(system_prompt, user_prompt)
        except Exception as exc:
            logger.warning("LLM %s failed: %s", label, exc)
            return None

    def suggest_column_mappings(
        self,
        proposal: MappingProposal,
        *,
        sheet: str,
    ) -> tuple[ColumnMapping, ...]:
        if not self.enabled:
            return ()

        low_confidence = proposal.low_confidence_mappings
        if not low_confidence:
            return ()

        headers = [mapping.source_column for mapping in low_confidence]
        canonical = _SHEET_CANONICAL_COLUMNS.get(sheet, CANONICAL_LITHOLOGY_COLUMNS)
        payload = self._complete_json(
            "You map spreadsheet headers only. Never invent data values.",
            (
                "Map spreadsheet headers to canonical borehole schema columns. "
                f"Headers: {headers}. Canonical options: {canonical}. "
                'Return JSON: {"mappings": [{"source": str, "canonical": str, "confidence": float}]}'
            ),
            label="column mapping",
        )
        if not payload:
            return ()
        return tuple(
            ColumnMapping(
                source_column=str(item["source"]),
                canonical_column=str(item["canonical"]),
                confidence=float(item.get("confidence", 0.5)),
            )
            for item in payload.get("mappings", [])
        )

    def suggest_lithology_mappings(self, unmapped_codes: Sequence[str]) -> tuple[LithologySuggestion, ...]:
        if not unmapped_codes:
            return ()

        local = _local_lithology_suggestions(unmapped_codes)
        if not self.enabled:
            return local

        payload = self._complete_json(
            "Suggest lithology normalizations only. Do not invent borehole data. "
            "Prefer Sand, Clay, Topsoil, Silt, Gravel when appropriate.",
            (
                "Suggest canonical lithology codes for field log descriptors. "
                f"Codes: {list(unmapped_codes)}. Options: {list(LITHOLOGY_SUGGESTION_OPTIONS)}. "
                'Return JSON: {"suggestions": [{"source": str, "canonical": str, '
                '"confidence": float, "rationale": str}]}'
            ),
            label="lithology mapping",
        )
        if not payload:
            return local

        suggestions = tuple(
            LithologySuggestion(
                source_code=str(item["source"]),
                canonical_code=canonical,
                confidence=float(item.get("confidence", 0.5)),
                rationale=str(item.get("rationale", "")),
            )
            for item in payload.get("suggestions", [])
            if (canonical := str(item["canonical"])) in _LITHOLOGY_OPTION_SET
        )
        return suggestions or local

    def explain_quality_issues(self, issues: Sequence[QualityIssue]) -> str:
        if not issues:
            return "No data quality issues were detected."

        local = _local_issue_summary(issues)
        if not self.enabled:
            return local

        payload = self._complete_json(
            "You summarize QA findings factually for engineering reports.",
            (
                "Write one concise paragraph summarizing these data quality findings for a geotechnical report. "
                "Use only the facts provided; do not invent new issues. "
                f"Facts: {json.dumps(_issue_facts(issues))}. Return JSON: {{\"summary\": str}}"
            ),
            label="narrative",
        )
        if not payload:
            return local
        return str(payload.get("summary", local))

    def suggest_fix_plan(self, issues: Sequence[QualityIssue]) -> tuple[FixStep, ...]:
        """Structured fix steps per issue code (local catalog; optional LLM polish)."""
        local = _local_fix_plan(issues)
        if not issues or not self.enabled:
            return local

        payload = self._complete_json(
            "You coach users to fix borehole QA issues. Never invent new issues or data.",
            (
                "Build a fix plan for borehole data QA issues. Use only the facts provided. "
                "For each issue return one step with action_id from: "
                "auto_unit_order, manual_unit_order, relative_elevation, manual_intervals, "
                "manual_hole_id, review_gaps, adjust_transect, add_lithology, check_coordinates, info_only. "
                f"Facts: {json.dumps(_issue_facts(issues))}. "
                'Return JSON: {"steps": [{"issue_code": str, "hole_id": str|null, '
                '"summary": str, "blocks_generate": bool, "action": str, "action_id": str}]}'
            ),
            label="fix plan",
        )
        if not payload:
            return local

        steps = tuple(
            FixStep(
                issue_code=code,
                hole_id=str(hole_id) if hole_id else None,
                summary=str(item.get("summary", code)),
                blocks_generate=bool(item.get("blocks_generate", code in _BLOCKS_GENERATE)),
                action=str(item.get("action", "")),
                action_id=str(item.get("action_id", "info_only")),
            )
            for item in payload.get("steps", [])
            for code in (str(item.get("issue_code", "")),)
            for hole_id in (item.get("hole_id"),)
        )
        return steps or local

    def suggest_report_metadata(self, context: dict) -> ReportMetadataSuggestion:
        """Propose consulting title-block / NOTES fields from section facts only."""
        local = _local_report_metadata(context)
        if not self.enabled:
            return local

        payload = self._complete_json(
            "You draft report title-block text for engineering figures. Facts only.",
            (
                "Propose consulting cross-section report metadata from these facts only. "
                "Do not invent geology or measurements. "
                f"Facts: {json.dumps(context)}. "
                'Return JSON: {"section_label": str, "map_scale": str, "notes": [str], '
                '"figure_caption": str, "prepared_for": str, "prepared_by": str, '
                '"source": str, "project_number": str, '
                '"transect_start_label": str, "transect_end_label": str}'
            ),
            label="report metadata",
        )
        if not payload:
            return local

        notes_raw = payload.get("notes") or list(local.notes)
        notes = tuple(str(item).strip() for item in notes_raw if str(item).strip())
        return ReportMetadataSuggestion(
            section_label=str(payload.get("section_label") or local.section_label),
            map_scale=str(payload.get("map_scale") or local.map_scale),
            notes=notes or local.notes,
            figure_caption=str(payload.get("figure_caption") or local.figure_caption),
            prepared_for=str(payload.get("prepared_for") or local.prepared_for),
            prepared_by=str(payload.get("prepared_by") or local.prepared_by),
            source=str(payload.get("source") or local.source),
            project_number=str(payload.get("project_number") or local.project_number),
            transect_start_label=str(
                payload.get("transect_start_label") or local.transect_start_label
            ),
            transect_end_label=str(
                payload.get("transect_end_label") or local.transect_end_label
            ),
        )

    def suggest_correlation_overrides(
        self,
        pair_summaries: Sequence[object],
        lithologies: Sequence[Lithology],
        hole_order: Sequence[str],
    ) -> tuple[CorrelationSuggestion, ...]:
        """Suggest CorrelationOverride pairs from interval tables (never invent depths)."""
        local = _local_correlation_suggestions(pair_summaries, lithologies, hole_order)
        if not self.enabled:
            return local

        hole_set = set(hole_order)
        intervals = [
            {
                "hole_id": lith.hole_id,
                "from_depth": lith.from_depth,
                "to_depth": lith.to_depth,
                "lithology_code": lith.lithology_code,
                "unit_order": lith.unit_order,
            }
            for lith in lithologies
            if lith.hole_id in hole_set
        ]
        pairs = [
            {
                "left_hole_id": getattr(summary, "left_hole_id"),
                "right_hole_id": getattr(summary, "right_hole_id"),
                "matched_count": getattr(summary, "matched_count"),
                "left_only_codes": list(getattr(summary, "left_only_codes", ())),
                "right_only_codes": list(getattr(summary, "right_only_codes", ())),
                "match_rate": float(getattr(summary, "match_rate", 0.0)),
            }
            for summary in pair_summaries
        ]
        payload = self._complete_json(
            "You suggest unit_order pairings only from provided intervals.",
            (
                "Suggest correlation overrides between adjacent holes. "
                "Only use unit_order values present in the interval table. "
                "Never invent depths or lithology codes. "
                f"Hole order: {list(hole_order)}. Pairs: {json.dumps(pairs)}. "
                f"Intervals: {json.dumps(intervals)}. "
                'Return JSON: {"suggestions": [{"left_hole_id": str, "right_hole_id": str, '
                '"left_unit_order": int, "right_unit_order": int, '
                '"confidence": float, "rationale": str}]}'
            ),
            label="correlation assist",
        )
        if not payload:
            return local

        orders_by_hole = _unit_orders_by_hole(lithologies)
        suggestions = tuple(
            CorrelationSuggestion(
                left_hole_id=left_id,
                right_hole_id=right_id,
                left_unit_order=left_order,
                right_unit_order=right_order,
                confidence=float(item.get("confidence", 0.5)),
                rationale=str(item.get("rationale", "")),
            )
            for item in payload.get("suggestions", [])
            for left_id in (str(item["left_hole_id"]),)
            for right_id in (str(item["right_hole_id"]),)
            for left_order in (int(item["left_unit_order"]),)
            for right_order in (int(item["right_unit_order"]),)
            if left_order in orders_by_hole.get(left_id, ())
            and right_order in orders_by_hole.get(right_id, ())
        )
        return suggestions or local

    def answer_section_question(self, question: str, facts: dict) -> str:
        """Grounded Q&A over active-section facts only."""
        question = question.strip()
        if not question:
            return "Ask a question about the active transect (holes, water, lithology, offsets)."

        local = _local_section_answer(question, facts)
        if not self.enabled:
            return local

        payload = self._complete_json(
            "You answer questions about a borehole cross-section using provided facts only.",
            (
                "Answer the user question using only these section facts. "
                "If the facts do not contain the answer, say you do not know. "
                "Do not invent geology. "
                f"Facts: {json.dumps(facts)}. Question: {question}. "
                'Return JSON: {"answer": str}'
            ),
            label="section Q&A",
        )
        if not payload:
            return local
        return str(payload.get("answer", "")).strip() or local

    def suggest_sheet_roles(
        self,
        sheet_names: Sequence[str],
        headers_by_sheet: dict[str, Sequence[str]],
    ) -> tuple[SheetRoleSuggestion, ...]:
        """Propose roles for workbook sheets (collars, lithology, water, screens, …)."""
        local = _local_sheet_roles(sheet_names, headers_by_sheet)
        if not self.enabled:
            return local

        payload = self._complete_json(
            "You classify spreadsheet sheets for borehole import. Never invent cell values.",
            (
                "Assign each workbook sheet a role. Roles: "
                f"{list(SHEET_ROLE_OPTIONS)}. "
                f"Sheets: {list(sheet_names)}. "
                f"Headers: {json.dumps({name: list(headers) for name, headers in headers_by_sheet.items()})}. "
                'Return JSON: {"suggestions": [{"sheet_name": str, "role": str, '
                '"confidence": float, "rationale": str}]}'
            ),
            label="sheet roles",
        )
        if not payload:
            return local

        allowed = set(SHEET_ROLE_OPTIONS)
        suggestions = tuple(
            SheetRoleSuggestion(
                sheet_name=str(item["sheet_name"]),
                role=role if role in allowed else "unknown",
                confidence=float(item.get("confidence", 0.5)),
                rationale=str(item.get("rationale", "")),
            )
            for item in payload.get("suggestions", [])
            for role in (str(item.get("role", "unknown")).lower(),)
        )
        return suggestions or local

    def parse_transect_request(
        self,
        text: str,
        available_hole_ids: Sequence[str],
    ) -> TransectParseResult | None:
        """Parse natural-language hole sequence; geometry still comes from collars."""
        text = text.strip()
        if not text or not available_hole_ids:
            return None

        local = _local_transect_parse(text, available_hole_ids)
        if not self.enabled:
            return local

        payload = self._complete_json(
            "You parse transect requests. Never invent hole IDs.",
            (
                "Extract an ordered borehole sequence and optional section label from the request. "
                f"Available hole IDs: {list(available_hole_ids)}. Request: {text}. "
                "Only use hole IDs from the available list. "
                'Return JSON: {"hole_ids": [str], "section_label": str, "rationale": str}'
            ),
            label="transect parse",
        )
        if not payload:
            return local

        available_lower = {hole_id.lower(): hole_id for hole_id in available_hole_ids}
        ordered: list[str] = []
        seen: set[str] = set()
        for item in payload.get("hole_ids", []):
            resolved = available_lower.get(str(item).strip().lower())
            if resolved and resolved not in seen:
                seen.add(resolved)
                ordered.append(resolved)
        if len(ordered) < 2:
            return local
        return TransectParseResult(
            hole_ids=tuple(ordered),
            section_label=str(
                payload.get("section_label") or (local.section_label if local else "")
            ),
            rationale=str(payload.get("rationale") or ""),
        )


def _issue_facts(issues: Sequence[QualityIssue]) -> list[dict[str, object]]:
    return [
        {
            "code": issue.code,
            "severity": issue.severity,
            "hole_id": issue.hole_id,
            "message": issue.message,
        }
        for issue in issues
    ]


def _local_issue_summary(issues: Sequence[QualityIssue]) -> str:
    errors = sum(1 for issue in issues if issue.severity == "error")
    warnings = sum(1 for issue in issues if issue.severity == "warning")
    infos = sum(1 for issue in issues if issue.severity == "info")
    return (
        f"Data quality review found {errors} error(s), {warnings} warning(s), "
        f"and {infos} informational item(s)."
    )


def _local_fix_plan(issues: Sequence[QualityIssue]) -> tuple[FixStep, ...]:
    steps: list[FixStep] = []
    seen: set[tuple[str, str | None]] = set()
    for issue in issues:
        key = (issue.code, issue.hole_id)
        if key in seen:
            continue
        seen.add(key)
        catalog = _FIX_CATALOG.get(issue.code)
        if catalog is None:
            steps.append(
                FixStep(
                    issue_code=issue.code,
                    hole_id=issue.hole_id,
                    summary=issue.message,
                    blocks_generate=issue.severity == "error",
                    action="Review the issue message and correct the source workbook.",
                    action_id="info_only",
                )
            )
            continue
        action_id, summary, action = catalog
        steps.append(
            FixStep(
                issue_code=issue.code,
                hole_id=issue.hole_id,
                summary=summary if not issue.hole_id else f"{summary} ({issue.hole_id})",
                blocks_generate=issue.code in _BLOCKS_GENERATE or issue.severity == "error",
                action=action,
                action_id=action_id,
            )
        )
    return tuple(steps)


def _local_lithology_suggestions(
    unmapped_codes: Sequence[str],
) -> tuple[LithologySuggestion, ...]:
    suggestions: list[LithologySuggestion] = []
    seen: set[str] = set()
    for source in unmapped_codes:
        if source in seen:
            continue
        seen.add(source)
        text = source.strip().lower()
        if text in {"ts", "top soil"}:
            suggestions.append(
                LithologySuggestion(source, "Topsoil", 0.9, "Heuristic keyword match")
            )
            continue
        if text == "cl":
            suggestions.append(
                LithologySuggestion(source, "Clay", 0.88, "Heuristic keyword match")
            )
            continue
        for needles, canonical, confidence in _LITHOLOGY_RULES:
            if any(needle in text for needle in needles):
                if canonical in CANONICAL_LITHOLOGY_CODES:
                    suggestions.append(
                        LithologySuggestion(
                            source_code=source,
                            canonical_code=canonical,
                            confidence=confidence,
                            rationale="Heuristic keyword match",
                        )
                    )
                break
    return tuple(suggestions)


def _local_report_metadata(context: dict) -> ReportMetadataSuggestion:
    hole_ids = [str(item) for item in context.get("hole_ids", [])]
    start = str(context.get("transect_start_label") or (hole_ids[0] if hole_ids else "A"))
    end = str(context.get("transect_end_label") or (hole_ids[-1] if hole_ids else "A'"))
    label = strip_cross_section_prefix(str(context.get("section_label") or "")) or "A-A'"

    notes = list(DEFAULT_CONSULTING_NOTES)
    water_count = int(context.get("water_measurement_count", 0))
    nm_holes = [str(item) for item in context.get("nm_hole_ids", [])]
    if nm_holes:
        notes.append(f"NM (not measured) at: {', '.join(nm_holes)}.")
    if water_count:
        notes.append(f"Groundwater elevations from {water_count} monitoring well observation(s).")
    ve = context.get("vertical_exaggeration")
    map_scale = str(context.get("map_scale") or "1:1000")
    caption_parts = [f"Cross section {label}"]
    if hole_ids:
        caption_parts.append(f"through {' → '.join(hole_ids)}")
    if ve is not None:
        try:
            caption_parts.append(f"({float(ve):g}× vertical exaggeration)")
        except (TypeError, ValueError):
            pass
    workbook_name = str(context.get("workbook_name") or "").strip()

    return ReportMetadataSuggestion(
        section_label=label,
        map_scale=map_scale,
        notes=tuple(notes),
        figure_caption=" ".join(caption_parts) + ".",
        prepared_for=str(context.get("prepared_for") or ""),
        prepared_by=str(context.get("prepared_by") or ""),
        source=workbook_name,
        project_number=str(context.get("project_number") or ""),
        transect_start_label=start if len(start) <= 8 else start[:1],
        transect_end_label=end if len(end) <= 8 else f"{end[:1]}'",
    )


def _unit_orders_by_hole(lithologies: Sequence[Lithology]) -> dict[str, frozenset[int]]:
    orders: dict[str, set[int]] = {}
    for lith in lithologies:
        if lith.unit_order is None:
            continue
        orders.setdefault(lith.hole_id, set()).add(lith.unit_order)
    return {hole_id: frozenset(values) for hole_id, values in orders.items()}


def _local_correlation_suggestions(
    pair_summaries: Sequence[object],
    lithologies: Sequence[Lithology],
    hole_order: Sequence[str],
) -> tuple[CorrelationSuggestion, ...]:
    by_hole: dict[str, list[Lithology]] = {}
    for lith in lithologies:
        by_hole.setdefault(lith.hole_id, []).append(lith)

    suggestions: list[CorrelationSuggestion] = []
    seen_links: set[tuple[str, str, int, int]] = set()
    for summary in pair_summaries:
        left_id = str(getattr(summary, "left_hole_id"))
        right_id = str(getattr(summary, "right_hole_id"))
        match_rate = float(getattr(summary, "match_rate", 1.0))
        if match_rate >= 0.99:
            continue
        left_intervals = sorted(
            by_hole.get(left_id, []),
            key=lambda item: (item.from_depth, item.to_depth),
        )
        right_intervals = sorted(
            by_hole.get(right_id, []),
            key=lambda item: (item.from_depth, item.to_depth),
        )
        # Pair same lithology codes by depth rank when both have unit_order.
        right_by_code: dict[str, list[Lithology]] = {}
        for interval in right_intervals:
            if interval.unit_order is None:
                continue
            right_by_code.setdefault(interval.lithology_code, []).append(interval)
        used_right: set[int] = set()
        for left in left_intervals:
            if left.unit_order is None:
                continue
            candidates = right_by_code.get(left.lithology_code, [])
            for right in candidates:
                if right.unit_order in used_right:
                    continue
                if right.unit_order == left.unit_order:
                    # Already natural match; skip override noise.
                    used_right.add(right.unit_order)
                    break
                link = (left_id, right_id, int(left.unit_order), int(right.unit_order))
                if link in seen_links:
                    used_right.add(right.unit_order)
                    break
                seen_links.add(link)
                suggestions.append(
                    CorrelationSuggestion(
                        left_hole_id=left_id,
                        right_hole_id=right_id,
                        left_unit_order=link[2],
                        right_unit_order=link[3],
                        confidence=0.7,
                        rationale=(
                            f"Pair {left.lithology_code} by depth rank "
                            f"({left.unit_order} ↔ {right.unit_order})"
                        ),
                    )
                )
                used_right.add(right.unit_order)
                break
    # Preserve hole_order adjacency only
    adjacent = {
        (hole_order[index], hole_order[index + 1])
        for index in range(len(hole_order) - 1)
    }
    return tuple(
        item
        for item in suggestions
        if (item.left_hole_id, item.right_hole_id) in adjacent
    )


def _local_section_answer(question: str, facts: dict) -> str:
    text = question.lower()
    hole_ids = [str(item) for item in facts.get("hole_ids", [])]
    water = facts.get("water_levels") or {}
    nm_holes = [str(item) for item in facts.get("nm_hole_ids", [])]
    thicknesses = facts.get("lithology_thicknesses") or {}
    offsets = facts.get("offsets_m") or {}
    warnings = [str(item) for item in facts.get("overlap_warnings", [])]

    if "nm" in text or "not measured" in text or "dry" in text:
        if nm_holes:
            return f"Wells without water measurements (NM): {', '.join(nm_holes)}."
        return "All active wells have water measurements in the current subset."
    if "water" in text or "groundwater" in text or "gw" in text:
        if not water:
            return "No water levels are loaded for the active transect."
        parts: list[str] = []
        for hole, depth_value in water.items():
            if isinstance(depth_value, dict):
                for series_id, depth in depth_value.items():
                    label = series_id if series_id != "default" else "default"
                    parts.append(f"{hole} ({label}) at {depth} m depth")
            else:
                parts.append(f"{hole} at {depth_value} m depth")
        return "Water levels: " + "; ".join(parts) + "."
    if "hole" in text or "well" in text or "which" in text and "transect" in text:
        if hole_ids:
            return "Active transect holes: " + " → ".join(hole_ids) + "."
        return "No holes are selected on the active transect."
    if "offset" in text:
        if not offsets:
            return "No offset distances are available for the active transect."
        parts = [f"{hole} {value:.1f} m" for hole, value in offsets.items()]
        return "Offsets from transect: " + "; ".join(parts) + "."
    if "thickness" in text or "litholog" in text or "clay" in text or "sand" in text:
        if not thicknesses:
            return "No lithology thickness facts are available."
        # Prefer mentioning a code named in the question.
        for code, by_hole in thicknesses.items():
            if code.lower() in text:
                parts = [f"{hole}: {thick} m" for hole, thick in by_hole.items()]
                return f"{code} thickness — " + "; ".join(parts) + "."
        lines = []
        for code, by_hole in list(thicknesses.items())[:6]:
            parts = [f"{hole} {thick} m" for hole, thick in by_hole.items()]
            lines.append(f"{code}: " + ", ".join(parts))
        return "Lithology thicknesses: " + " | ".join(lines) + "."
    if "overlap" in text or "warning" in text:
        if warnings:
            return "Overlap warnings: " + "; ".join(warnings)
        return "No polygon overlap warnings are recorded for this section."
    if hole_ids:
        return (
            "I can answer from active-section facts only "
            f"(holes: {' → '.join(hole_ids)}). Ask about water, NM, offsets, or thicknesses."
        )
    return "No active-section facts are loaded."


def _local_sheet_roles(
    sheet_names: Sequence[str],
    headers_by_sheet: dict[str, Sequence[str]],
) -> tuple[SheetRoleSuggestion, ...]:
    from ai_quality import SHEET_ALIASES

    collar_aliases = SHEET_ALIASES.get("collars", set())
    lithology_aliases = SHEET_ALIASES.get("lithology", set())
    suggestions: list[SheetRoleSuggestion] = []
    for name in sheet_names:
        key = name.strip().lower().replace(" ", "_")
        key_compact = key.replace("_", "")
        name_tokens = {key, key_compact, name.strip().lower()}
        headers = {str(item).strip().lower() for item in headers_by_sheet.get(name, [])}
        role = "unknown"
        confidence = 0.4
        rationale = "No strong header match"
        if name_tokens & collar_aliases or (
            {"easting", "northing"} <= headers or {"lat", "long"} <= headers or {"latitude", "longitude"} <= headers
        ):
            role = "collars"
            confidence = 0.85
            rationale = "Name or coordinate headers"
        elif name_tokens & lithology_aliases or (
            ({"from_depth", "to_depth"} <= headers or {"from", "to"} <= headers)
            and "lithology" in " ".join(headers)
        ):
            role = "lithology"
            confidence = 0.85
            rationale = "Depth interval headers"
        elif key in {"water", "gw", "groundwater"} or (
            "depth" in headers and "hole_id" in headers and "from_depth" not in headers
        ):
            role = "water"
            confidence = 0.8
            rationale = "Water-level style headers"
        elif key in {"screens", "screen"}:
            role = "screens"
            confidence = 0.9
            rationale = "Sheet name"
        elif key in {"gradients", "gradient"}:
            role = "gradients"
            confidence = 0.9
            rationale = "Sheet name"
        elif key in {"deviations", "deviation", "survey"}:
            role = "deviations"
            confidence = 0.85
            rationale = "Sheet name"
        elif {"from_depth", "to_depth", "lithology_code"} <= headers or {
            "from_depth",
            "to_depth",
            "lithology",
        } <= headers:
            role = "lithology"
            confidence = 0.8
            rationale = "Interval + lithology headers"
        suggestions.append(
            SheetRoleSuggestion(
                sheet_name=name,
                role=role,
                confidence=confidence,
                rationale=rationale,
            )
        )
    return tuple(suggestions)


def _local_transect_parse(
    text: str,
    available_hole_ids: Sequence[str],
) -> TransectParseResult | None:
    if not available_hole_ids:
        return None
    available_lower = {hole_id.lower(): hole_id for hole_id in available_hole_ids}
    # Single alternation, longest IDs first to avoid partial matches.
    keys = sorted(available_lower, key=len, reverse=True)
    pattern = re.compile(
        r"(?<![a-z0-9])(" + "|".join(re.escape(key) for key in keys) + r")(?![a-z0-9])",
        flags=re.IGNORECASE,
    )
    found: list[str] = []
    seen: set[str] = set()
    for match in pattern.finditer(text):
        resolved = available_lower[match.group(1).lower()]
        if resolved not in seen:
            seen.add(resolved)
            found.append(resolved)
    if len(found) < 2:
        return None

    label = ""
    label_match = _TRANSECT_LABEL_RE.search(text)
    if label_match:
        label = (label_match.group(1) or label_match.group(2) or "").upper()
        if len(label) == 1:
            label = f"{label}-{label}'"
    return TransectParseResult(
        hole_ids=tuple(found),
        section_label=label,
        rationale="Parsed hole IDs in order of appearance",
    )
