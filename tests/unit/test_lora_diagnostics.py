from pathlib import Path

from llm4rec.evaluation.lora_diagnostics import diagnose_lora_predictions
from llm4rec.io.artifacts import write_jsonl


def test_diagnose_lora_predictions_flags_prompt_continuation(tmp_path: Path):
    predictions = tmp_path / "predictions.jsonl"
    write_jsonl(
        predictions,
        [
            {
                "candidate_items": ["i1", "i2"],
                "dataset": "tiny",
                "metadata": {"parse_success": True},
                "method": "local_8b_lora::history_only_sft",
                "predicted_items": ["i2", "i1"],
                "raw_output": "2\nuser: Rank candidate item IDs\nCandidates: ['i1', 'i2']",
                "target_item": "i2",
                "user_id": "u1",
            },
            {
                "candidate_items": ["i1", "i3"],
                "dataset": "tiny",
                "metadata": {"parse_success": False},
                "method": "local_8b_lora::history_only_sft",
                "predicted_items": ["i1", "i3"],
                "raw_output": "",
                "target_item": "i3",
                "user_id": "u2",
            },
        ],
    )

    result = diagnose_lora_predictions(predictions, tmp_path / "diagnostics")

    assert result["overall"]["num_rows"] == 2
    assert result["overall"]["prompt_continuation_rate"] == 0.5
    assert result["overall"]["target_at_last_candidate_rate"] == 1.0
    assert (tmp_path / "diagnostics" / "lora_prediction_diagnostics.csv").is_file()
