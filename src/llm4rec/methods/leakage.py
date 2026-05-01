"""Leakage validation for time-aware graph evidence methods."""

from __future__ import annotations

from typing import Any

from llm4rec.evidence.base import Evidence


class LeakageViolation(ValueError):
    """Raised when a reportable run would leak future or label information."""


class LeakageValidator:
    """Validate examples, evidence, prompts, and configs for leakage risks."""

    def __init__(self, *, reportable: bool = False) -> None:
        self.reportable = bool(reportable)

    def validate_example(
        self,
        *,
        history: list[str],
        target_item: str | None,
        candidate_items: list[str],
        prediction_timestamp: float | None = None,
        candidate_metadata: dict[str, Any] | None = None,
    ) -> None:
        """Validate a ranking example."""

        del prediction_timestamp
        target = None if target_item is None else str(target_item)
        if target is not None and target in {str(item) for item in history}:
            raise LeakageViolation("user history contains the target item")
        metadata = dict(candidate_metadata or {})
        if self.reportable and metadata.get("target_derived_future_info"):
            raise LeakageViolation("candidate set contains target-derived future information")
        if self.reportable and not candidate_items:
            raise LeakageViolation("reportable runs require an explicit candidate set")

    def validate_prompt(self, *, prompt: str, target_item: str | None) -> None:
        """Validate that the LLM prompt does not expose the target label."""

        if target_item is None:
            return
        token = str(target_item)
        if token and token in str(prompt):
            raise LeakageViolation("LLM prompt includes the target label")

    def validate_evidence(
        self,
        evidence: Evidence,
        *,
        prediction_timestamp: float | None = None,
    ) -> None:
        """Validate a single evidence object."""

        provenance = evidence.provenance
        if not provenance:
            raise LeakageViolation("evidence provenance is required")
        constructed_from = provenance.get("constructed_from")
        if constructed_from not in {"train_only", "diagnostic_only"}:
            raise LeakageViolation("evidence constructed_from must be train_only or diagnostic_only")
        if self.reportable and constructed_from != "train_only":
            raise LeakageViolation("reportable runs may only use train_only evidence")
        if self.reportable and str(provenance.get("split", "")).lower() != "train":
            raise LeakageViolation("reportable evidence graph must be constructed from train split")
        target_timestamp = evidence.timestamp_info.get("target_timestamp")
        if (
            prediction_timestamp is not None
            and target_timestamp is not None
            and float(target_timestamp) > float(prediction_timestamp)
        ):
            raise LeakageViolation("time-window evidence uses an event after prediction timestamp")

    def validate_config(self, config: dict[str, Any]) -> None:
        """Validate method or experiment config safety."""

        reportable = self.reportable or bool(
            config.get("reportable", config.get("method", {}).get("reportable", False))
        )
        llm = dict(config.get("llm", {}))
        method = dict(config.get("method", {}))
        encoder = dict(config.get("encoder", {}))
        method_name = str(method.get("name", config.get("name", ""))).lower()
        provider = str(llm.get("provider", "")).lower()
        encoder_type = str(encoder.get("type", "")).lower()
        if reportable and provider == "mock":
            raise LeakageViolation("mock provider is blocked in reportable configs")
        if reportable and "stub" in encoder_type:
            raise LeakageViolation("stub encoder is blocked in reportable configs")
        if reportable and any(token in method_name for token in ("smoke", "skeleton", "markov")):
            raise LeakageViolation("smoke/skeleton/Markov methods must be reportable=false")
        if reportable and config.get("constructed_from") == "diagnostic_only":
            raise LeakageViolation("diagnostic-only artifacts are blocked in reportable configs")

    def validate_encoder_training_events(
        self,
        *,
        events: list[dict[str, Any]],
        constructed_from: str,
        prediction_timestamp: float | None = None,
    ) -> None:
        """Validate dynamic encoder events do not include future evidence."""

        if self.reportable and constructed_from != "train_only":
            raise LeakageViolation("reportable dynamic encoders must train on train_only events")
        if prediction_timestamp is None:
            return
        for event in events:
            timestamp = event.get("timestamp")
            if timestamp is not None and float(timestamp) > float(prediction_timestamp):
                raise LeakageViolation("dynamic encoder event occurs after prediction timestamp")


def validate_evidence_list(
    evidence: list[Evidence],
    *,
    reportable: bool,
    prediction_timestamp: float | None = None,
) -> None:
    """Validate a list of evidence rows."""

    validator = LeakageValidator(reportable=reportable)
    for row in evidence:
        validator.validate_evidence(row, prediction_timestamp=prediction_timestamp)
