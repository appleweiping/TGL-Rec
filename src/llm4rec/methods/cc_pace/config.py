"""Configuration for CC-PACE.

A single frozen dataclass captures every knob the method exposes, so configs are
reproducible and every hyperparameter is sweepable for the paper's
hyperparameter-stability experiment (see docs/paper_followup_experiments.md).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CCPaceConfig:
    """All CC-PACE hyperparameters and structural switches.

    The ``use_*`` switches exist so the ablation harness can disable each
    component independently (ablation-by-design, docs/method_v2_decision_CC-PACE.md
    section 5). Disabling a component must hurt on its designated slice; if it
    does not, the component is badly designed and should be reported honestly.
    """

    # --- identity ---
    name: str = "cc_pace"
    codename: str = "CC-PACE"
    eval_split: str = "test"

    # --- panel / schema ---
    max_history_items: int = 20
    randomize_label_ids: bool = True          # exchangeability requirement
    n_label_randomizations: int = 4           # R: averaged evidence + uncertainty
    schema_fields: tuple[str, ...] = ("category", "brand", "keywords", "attrs")
    use_profile_slots: bool = True            # long-term profile (beauty SOTA needs this)
    profile_slot_keys: tuple[str, ...] = (
        "top_categories", "liked_brands", "concerns", "routine_step",
        "ingredient_prefs", "price_band",
    )

    # --- collaborative conditioning (frozen CF; sigma-field, not a score head) ---
    use_cf_tokens: bool = True                # rendered neighbour evidence in prompt
    use_cf_nuisance: bool = True              # r_CF / CF_emb / CF_cluster in residualizer
    cf_model: str = "sasrec"                  # frozen; provided externally
    cf_num_neighbors: int = 10

    # --- residualizer (symmetric LOO) ---
    use_residualizer: bool = True
    residualizer_kind: str = "isotonic"       # {"isotonic","gam","none"}
    residualizer_rich: bool = False           # m_hat_rich ceiling test (tensor GAM / GBM)
    residual_nuisance_keys: tuple[str, ...] = (
        "content_emb", "facet_bucket", "log_pop", "cf_emb", "r_cf", "cf_cluster",
    )

    # --- dual null + conformal (calibration / abstention layer only) ---
    use_dual_null: bool = True                # P_sem auxiliary reference panel
    n_sem_nulls: int = 100
    conformal_combine: str = "max"            # {"max" (IUT), "evalue_avg"}
    split_conformal: bool = True              # train fold A, calibrate fold B

    # --- sparse-user shrinkage (empirical-Bayes James-Stein) ---
    use_shrinkage: bool = True
    shrinkage_n_cap: int = 20

    # --- popularity-orthogonality penalty (training-time) ---
    use_dcor_penalty: bool = True
    dcor_weight: float = 0.1

    # --- training (judge LoRA, listwise Plackett-Luce) ---
    train_loss: str = "plackett_luce"         # {"plackett_luce","softmax_nll"}
    lora_rank: int = 16
    lora_alpha: int = 32
    learning_rate: float = 1e-4
    temperature: float = 1.0
    temperature_min: float = 0.05
    epochs: int = 2
    batch_panels: int = 8

    # --- backbone ---
    backbone_model: str = "/home/ajifang/models/Qwen/Qwen3-8B"
    max_context_tokens: int = 32768
    schema_tokens_per_item: int = 90

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


DEFAULT_CC_PACE_CONFIG = CCPaceConfig()


def load_cc_pace_config(path_or_config: "str | Path | dict[str, Any] | None") -> CCPaceConfig:
    """Build a CCPaceConfig from a dict / YAML path, falling back to defaults.

    Unknown keys are ignored (forward-compatible); known keys override defaults.
    """
    if path_or_config is None:
        return DEFAULT_CC_PACE_CONFIG
    if isinstance(path_or_config, (str, Path)):
        from llm4rec.experiments.config import load_yaml_config

        raw = load_yaml_config(path_or_config)
        raw = raw.get("cc_pace", raw) if isinstance(raw, dict) else {}
    else:
        raw = dict(path_or_config)
    fields = {f for f in DEFAULT_CC_PACE_CONFIG.to_dict()}
    overrides = {k: v for k, v in raw.items() if k in fields}
    return CCPaceConfig(**{**DEFAULT_CC_PACE_CONFIG.to_dict(), **overrides})
