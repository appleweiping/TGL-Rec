"""Config-driven ablation switches for TimeGraphEvidenceRec."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any


REQUIRED_ABLATIONS = [
    "full",
    "w_o_llm",
    "w_o_retrieval",
    "w_o_temporal_graph",
    "w_o_transition_edges",
    "w_o_time_window_edges",
    "w_o_time_gap_tags",
    "w_o_semantic_similarity",
    "w_o_user_profile",
    "w_o_grounding_constraint",
    "w_o_explanation",
    "w_o_dynamic_encoder",
    "encoder_only",
    "text_only",
    "graph_only",
]


@dataclass(frozen=True)
class AblationSwitches:
    """Boolean feature switches used to instantiate ablation variants."""

    use_llm: bool = False
    use_retrieval: bool = True
    use_temporal_graph: bool = True
    use_transition_edges: bool = True
    use_time_window_edges: bool = True
    use_time_gap_tags: bool = True
    use_semantic_similarity: bool = True
    use_user_profile: bool = True
    use_recent_block: bool = True
    use_long_term_block: bool = True
    use_grounding_constraint: bool = True
    use_explanation: bool = True
    use_dynamic_encoder: bool = False

    def to_dict(self) -> dict[str, bool]:
        return {field.name: bool(getattr(self, field.name)) for field in fields(self)}

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "AblationSwitches":
        known = {field.name for field in fields(cls)}
        unknown = sorted(set(values) - known)
        if unknown:
            raise ValueError(f"Unknown ablation switches: {unknown}")
        return cls(**{key: bool(value) for key, value in values.items()})


def ablation_switches(name: str) -> AblationSwitches:
    """Return switches for a named ablation."""

    base = AblationSwitches()
    overrides: dict[str, dict[str, bool]] = {
        "full": {},
        "w_o_llm": {"use_llm": False},
        "w_o_retrieval": {"use_retrieval": False},
        "w_o_temporal_graph": {
            "use_temporal_graph": False,
            "use_transition_edges": False,
            "use_time_window_edges": False,
            "use_time_gap_tags": False,
        },
        "w_o_transition_edges": {"use_transition_edges": False},
        "w_o_time_window_edges": {"use_time_window_edges": False},
        "w_o_time_gap_tags": {"use_time_gap_tags": False},
        "w_o_semantic_similarity": {"use_semantic_similarity": False},
        "w_o_user_profile": {
            "use_user_profile": False,
            "use_recent_block": False,
            "use_long_term_block": False,
        },
        "w_o_grounding_constraint": {"use_grounding_constraint": False},
        "w_o_explanation": {"use_explanation": False},
        "w_o_dynamic_encoder": {"use_dynamic_encoder": False},
        "encoder_only": {
            "use_llm": False,
            "use_retrieval": False,
            "use_temporal_graph": False,
            "use_transition_edges": False,
            "use_time_window_edges": False,
            "use_time_gap_tags": False,
            "use_semantic_similarity": False,
            "use_user_profile": False,
            "use_recent_block": False,
            "use_long_term_block": False,
            "use_grounding_constraint": False,
            "use_explanation": False,
            "use_dynamic_encoder": True,
        },
        "text_only": {
            "use_temporal_graph": False,
            "use_transition_edges": False,
            "use_time_window_edges": False,
            "use_time_gap_tags": False,
            "use_dynamic_encoder": False,
        },
        "graph_only": {
            "use_llm": False,
            "use_semantic_similarity": False,
            "use_explanation": False,
            "use_dynamic_encoder": False,
        },
    }
    if name not in overrides:
        raise ValueError(f"Unknown ablation name: {name}")
    values = base.to_dict()
    values.update(overrides[name])
    return AblationSwitches.from_dict(values)


def build_ablation_configs(
    method_config: dict[str, Any],
    *,
    names: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Create method configs for all requested ablations."""

    requested = names or REQUIRED_ABLATIONS
    output: list[dict[str, Any]] = []
    for name in requested:
        switches = ablation_switches(str(name))
        config = _deep_copy(method_config)
        config.setdefault("method", {})
        config["method"]["ablation_name"] = str(name)
        config["method"]["reportable"] = False
        config["ablation"] = switches.to_dict()
        output.append(config)
    return output


def validate_ablation_names(names: list[str]) -> None:
    """Validate that requested ablations are known."""

    missing = sorted(set(names) - set(REQUIRED_ABLATIONS))
    if missing:
        raise ValueError(f"Unknown ablation names: {missing}")


def _deep_copy(value: Any) -> Any:
    import copy

    return copy.deepcopy(value)
