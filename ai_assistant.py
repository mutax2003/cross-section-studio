"""Optional LLM assist for column mapping, lithology synonyms, and QA narratives."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Protocol, Sequence

from ai_quality import ColumnMapping, MappingProposal, QualityIssue

logger = logging.getLogger(__name__)

CANONICAL_COLLAR_COLUMNS = ["hole_id", "easting", "northing", "elevation", "total_depth"]
CANONICAL_LITHOLOGY_COLUMNS = ["hole_id", "from_depth", "to_depth", "lithology_code", "hatch_pattern"]
USGS_LITHOLOGY_CODES = ["Sandstone", "Clay", "Silt", "Gravel", "Bedrock", "Limestone", "Shale"]


@dataclass(frozen=True)
class LithologySuggestion:
    source_code: str
    canonical_code: str
    confidence: float
    rationale: str


class LLMProvider(Protocol):
    def complete_json(self, system_prompt: str, user_prompt: str) -> dict:
        ...


class OpenAIProvider:
    """Optional OpenAI-backed provider; disabled when API key is missing."""

    def __init__(self, api_key: str, model: str = "gpt-4o-mini") -> None:
        self.api_key = api_key
        self.model = model

    def complete_json(self, system_prompt: str, user_prompt: str) -> dict:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("openai package is not installed") from exc

        client = OpenAI(api_key=self.api_key)
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
        return json.loads(content)


class MockLLMProvider:
    """Deterministic provider for tests."""

    def __init__(self, responses: dict[str, dict] | None = None) -> None:
        self.responses = responses or {}

    def complete_json(self, system_prompt: str, user_prompt: str) -> dict:
        if "column" in user_prompt.lower():
            return self.responses.get(
                "columns",
                {
                    "mappings": [
                        {"source": "BH", "canonical": "hole_id", "confidence": 0.95},
                    ]
                },
            )
        if "lithology" in user_prompt.lower():
            return self.responses.get(
                "lithology",
                {
                    "suggestions": [
                        {
                            "source": "Fat Clay",
                            "canonical": "Clay",
                            "confidence": 0.92,
                            "rationale": "Common field log descriptor",
                        }
                    ]
                },
            )
        return self.responses.get(
            "narrative",
            {"summary": "Data quality review found depth gaps and one off-transect borehole."},
        )


class AIAssistant:
    """Hybrid assistant: local rules first, LLM suggestions when enabled."""

    def __init__(self, provider: LLMProvider | None = None) -> None:
        self.provider = provider

    @property
    def enabled(self) -> bool:
        return self.provider is not None

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
        canonical = CANONICAL_COLLAR_COLUMNS if sheet == "collars" else CANONICAL_LITHOLOGY_COLUMNS
        prompt = (
            "Map spreadsheet headers to canonical borehole schema columns. "
            f"Headers: {headers}. Canonical options: {canonical}. "
            "Return JSON: {\"mappings\": [{\"source\": str, \"canonical\": str, \"confidence\": float}]}"
        )
        try:
            payload = self.provider.complete_json(
                "You map spreadsheet headers only. Never invent data values.",
                prompt,
            )
        except Exception as exc:
            logger.warning("LLM column mapping failed: %s", exc)
            return ()

        suggestions: list[ColumnMapping] = []
        for item in payload.get("mappings", []):
            suggestions.append(
                ColumnMapping(
                    source_column=str(item["source"]),
                    canonical_column=str(item["canonical"]),
                    confidence=float(item.get("confidence", 0.5)),
                )
            )
        return tuple(suggestions)

    def suggest_lithology_mappings(self, unmapped_codes: Sequence[str]) -> tuple[LithologySuggestion, ...]:
        if not self.enabled or not unmapped_codes:
            return ()

        prompt = (
            "Suggest canonical USGS-style lithology codes for field log descriptors. "
            f"Codes: {list(unmapped_codes)}. Options: {USGS_LITHOLOGY_CODES}. "
            "Return JSON: {\"suggestions\": [{\"source\": str, \"canonical\": str, "
            "\"confidence\": float, \"rationale\": str}]}"
        )
        try:
            payload = self.provider.complete_json(
                "Suggest lithology normalizations only. Do not invent borehole data.",
                prompt,
            )
        except Exception as exc:
            logger.warning("LLM lithology mapping failed: %s", exc)
            return ()

        return tuple(
            LithologySuggestion(
                source_code=str(item["source"]),
                canonical_code=str(item["canonical"]),
                confidence=float(item.get("confidence", 0.5)),
                rationale=str(item.get("rationale", "")),
            )
            for item in payload.get("suggestions", [])
        )

    def explain_quality_issues(self, issues: Sequence[QualityIssue]) -> str:
        if not issues:
            return "No data quality issues were detected."

        if not self.enabled:
            return _local_issue_summary(issues)

        facts = [
            {
                "code": issue.code,
                "severity": issue.severity,
                "hole_id": issue.hole_id,
                "message": issue.message,
            }
            for issue in issues
        ]
        prompt = (
            "Write one concise paragraph summarizing these data quality findings for a geotechnical report. "
            "Use only the facts provided; do not invent new issues. "
            f"Facts: {json.dumps(facts)}. Return JSON: {{\"summary\": str}}"
        )
        try:
            payload = self.provider.complete_json(
                "You summarize QA findings factually for engineering reports.",
                prompt,
            )
            return str(payload.get("summary", _local_issue_summary(issues)))
        except Exception as exc:
            logger.warning("LLM narrative failed: %s", exc)
            return _local_issue_summary(issues)


def _local_issue_summary(issues: Sequence[QualityIssue]) -> str:
    errors = sum(1 for issue in issues if issue.severity == "error")
    warnings = sum(1 for issue in issues if issue.severity == "warning")
    infos = sum(1 for issue in issues if issue.severity == "info")
    return (
        f"Data quality review found {errors} error(s), {warnings} warning(s), "
        f"and {infos} informational item(s)."
    )
