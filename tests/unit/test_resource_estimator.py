from llm4rec.experiments.resource_estimator import estimate_pilot_resources


def test_resource_estimator_blocks_api_and_lora_budget():
    estimate = estimate_pilot_resources("configs/experiments/phase7_pilot_movielens_sample.yaml")
    assert estimate["estimated_api_calls"] == 0
    assert estimate["allow_api_calls"] is False
    assert estimate["enable_lora_training"] is False
    assert estimate["non_reportable_marker"] == "NON_REPORTABLE"
