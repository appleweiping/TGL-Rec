from pathlib import Path

from llm4rec.diagnostics.llm_sequence_time_runner import run_llm_sequence_time_diagnostics


def test_phase3a_mock_llm_diagnostics_run():
    run_dir = run_llm_sequence_time_diagnostics("configs/diagnostics/llm_sequence_time_mock.yaml")

    assert run_dir == Path("outputs/runs/phase3a_llm_diagnostics").resolve()
    assert (run_dir / "llm_requests.jsonl").is_file()
    assert (run_dir / "llm_raw_outputs.jsonl").is_file()
    assert (run_dir / "predictions.jsonl").is_file()
    assert (run_dir / "metrics.json").is_file()
    assert (run_dir / "cost_latency.json").is_file()
    assert (run_dir / "llm_diagnostic_summary.json").is_file()
    assert (run_dir / "diagnostics" / "prompt_variant_results.csv").is_file()
    assert (run_dir / "diagnostics" / "hallucination_cases.jsonl").is_file()
    assert (run_dir / "diagnostics" / "parse_failures.jsonl").is_file()

