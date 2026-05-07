from __future__ import annotations

from pathlib import Path

from llm4rec.experiments.four_domain_plan import (
    build_four_domain_server_plan,
    write_shell_runbook,
)


def test_four_domain_plan_detects_existing_and_missing_task_dirs(tmp_path: Path) -> None:
    external = tmp_path / "external_tasks"
    (external / "books_large10000_100neg_test_same_candidate").mkdir(parents=True)

    plan = build_four_domain_server_plan(
        external_root=external,
        domains=["books", "movies"],
        splits=["test"],
    )

    tasks = plan["planned_task_dirs"]
    assert tasks == [
        {
            "domain": "books",
            "exists": True,
            "path": str(external / "books_large10000_100neg_test_same_candidate"),
            "split": "test",
        },
        {
            "domain": "movies",
            "exists": False,
            "path": str(external / "movies_large10000_100neg_test_same_candidate"),
            "split": "test",
        },
    ]
    assert len(plan["missing_task_dirs"]) == 1
    assert "--task-dir" in plan["commands"]["import_frozen_same_candidate_tasks"][0]
    assert "books_large10000_100neg_test_same_candidate" in (
        plan["commands"]["import_frozen_same_candidate_tasks"][0]
    )
    assert "movies_large10000_100neg_test_same_candidate" not in (
        plan["commands"]["import_frozen_same_candidate_tasks"][0]
    )


def test_four_domain_plan_can_write_shell_runbook(tmp_path: Path) -> None:
    external = tmp_path / "external_tasks"
    (external / "beauty_large10000_100neg_valid_same_candidate").mkdir(parents=True)
    plan = build_four_domain_server_plan(
        external_root=external,
        domains=["beauty"],
        splits=["valid"],
    )
    output = write_shell_runbook(plan, tmp_path / "runbook.sh")

    text = output.read_text(encoding="utf-8")

    assert "set -euo pipefail" in text
    assert "git pull" in text
    assert "import_week8_same_candidate.py" in text
