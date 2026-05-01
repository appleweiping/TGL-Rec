import json
from pathlib import Path

from llm4rec.diagnostics.api_micro_runner import run_api_micro_diagnostic


def test_phase3b_api_micro_dry_run_without_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    run_dir = run_api_micro_diagnostic(
        "configs/diagnostics/llm_sequence_time_api_micro.yaml",
        dry_run=True,
    )
    root = Path(run_dir)
    preflight = json.loads((root / "cost_preflight.json").read_text(encoding="utf-8"))
    sampled = json.loads(
        (root / "artifacts" / "api_sampled_cases.json").read_text(encoding="utf-8")
    )
    summary = json.loads((root / "api_micro_summary.json").read_text(encoding="utf-8"))
    assert preflight["estimated_api_calls"] <= preflight["max_api_calls"]
    assert sampled["sample_count"] == preflight["number_of_cases"]
    assert summary["dry_run"] is True
    assert (root / "api_requests.jsonl").is_file()
    assert (root / "api_raw_outputs.jsonl").read_text(encoding="utf-8") == ""
    assert "not set" in " ".join(summary["guard_warnings"])
