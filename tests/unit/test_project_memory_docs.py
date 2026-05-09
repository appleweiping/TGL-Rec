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
    assert "cannot be stitched, copied, or presented as a recombination" in text
    assert "Large-Scale Observation Milestone" in text
    assert "Formal Baseline Milestone" in text
    assert "For every complex task, use multi-agent collaboration" in text
    assert "multiple top-conference papers or official projects" in text
    assert "basically complete and ready for paper writing" in text
    assert "At the end of every complex task" in text
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


def test_top_level_agent_rules_include_complex_task_protocol() -> None:
    agents = _read("AGENTS.md")
    phase10 = _read("docs/phase10_master_plan.md")

    assert "Use multi-agent collaboration for every complex task" in agents
    assert "multiple top-conference papers/projects" in agents
    assert "current gate toward ending experiments" in agents
    assert "Experiment Ending Gate" in phase10
    assert "Every complex-task final report" in phase10
