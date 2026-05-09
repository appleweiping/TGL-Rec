from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _read(relative_path: str) -> str:
    return (ROOT / relative_path).read_text(encoding="utf-8")


def test_project_memory_records_non_toy_phase10_direction() -> None:
    text = _read("docs/codex_project_memory.md")

    assert "Phase 10" in text
    assert "toy demo" in text
    assert "Qwen3-8B" in text
    assert "project LoRA/QLoRA regime" in text
    assert "official implementation" in text
    assert "multi-agent" in text
    assert "cannot inspect the shared server directly" in text
    for domain in ("beauty", "books", "electronics", "movies"):
        assert domain in text


def test_first_read_docs_point_to_project_memory() -> None:
    for relative_path in (
        "AGENTS.md",
        "README.md",
        "docs/phase10_master_plan.md",
        "docs/server_runbook.md",
        "docs/codex_handoff_phase9e.md",
        "docs/reportable_rules.md",
    ):
        assert "docs/codex_project_memory.md" in _read(relative_path)


def test_codex_agent_roles_include_phase10_memory() -> None:
    for relative_path in (
        ".codex/agents/research-worker.toml",
        ".codex/agents/reviewer.toml",
        ".codex/agents/repro-auditor.toml",
        ".codex/agents/lit-scout.toml",
        ".codex/agents/experiment-runner.toml",
    ):
        text = _read(relative_path)
        assert "docs/codex_project_memory.md" in text

