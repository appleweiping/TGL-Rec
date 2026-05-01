import pytest

from llm4rec.trainers.lora import LoRAConfigError, validate_lora_config


def _config():
    return {
        "training": {
            "adapter_output_dir": "outputs/test_lora",
            "base_model": "local",
            "batch_size": 1,
            "gradient_accumulation_steps": 2,
            "learning_rate": 0.0002,
            "lora_alpha": 16,
            "lora_dropout": 0.1,
            "lora_r": 8,
            "max_seq_length": 128,
            "num_train_epochs": 1,
            "quantization": "4bit",
            "save_steps": 10,
            "target_modules": ["q_proj"],
        }
    }


def test_lora_config_validation_accepts_required_fields():
    assert validate_lora_config(_config())["base_model"] == "local"


def test_lora_config_validation_rejects_missing_field():
    config = _config()
    del config["training"]["base_model"]
    with pytest.raises(LoRAConfigError):
        validate_lora_config(config)
