from llm4rec.evaluation.significance import paired_bootstrap_ci, paired_randomization_test


def test_significance_interfaces_handle_small_and_paired_samples():
    assert paired_randomization_test([1.0], [0.0])["warning"] == "insufficient_sample"
    result = paired_randomization_test([0.0, 1.0, 0.0], [1.0, 1.0, 0.0], num_rounds=20, seed=1)
    assert 0 <= result["p_value"] <= 1
    ci = paired_bootstrap_ci([0.0, 1.0, 0.0], [1.0, 1.0, 0.0], num_rounds=20, seed=1)
    assert ci["ci_low"] <= ci["ci_high"]
