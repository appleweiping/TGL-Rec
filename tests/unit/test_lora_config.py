import pytest

from llm4rec.trainers.lora_config import LoRA8BConfigError, load_lora_8b_sections


def _config():
    return {
        "model": {"base_model_path": "local/model", "max_seq_length": 2048},
        "training": {
            "lora_r": 16,
            "lora_alpha": 32,
            "target_modules": ["q_proj"],
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 4,
            "learning_rate": 2e-4,
        },
    }


def test_lora_config_validates_required_model_path():
    model, training = load_lora_8b_sections(_config())

    assert model.base_model_path == "local/model"
    assert training.lora_r == 16


def test_lora_config_rejects_missing_model_path():
    cfg = _config()
    cfg["model"].pop("base_model_path")

    with pytest.raises(LoRA8BConfigError):
        load_lora_8b_sections(cfg)
