from llm4rec.evaluation.significance import PairedMoments, paired_t_test, paired_t_test_from_moments


def test_paired_t_test_reports_direction_and_p_value():
    result = paired_t_test([0.0, 0.0, 1.0, 0.0], [1.0, 1.0, 1.0, 0.0])

    assert result["effect_direction"] == "method_a_better"
    assert result["n"] == 4
    assert result["p_value"] is not None


def test_paired_t_test_from_moments_handles_insufficient_samples():
    result = paired_t_test_from_moments(PairedMoments(n=1, sum_delta=1.0, sum_delta_sq=1.0))

    assert result["warning"] == "insufficient_sample"
    assert result["significant_at_0_05"] is False
