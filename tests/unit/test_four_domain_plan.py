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


def test_four_domain_plan_includes_week8_sft_merge_train_and_eval(tmp_path: Path) -> None:
    external = tmp_path / "external_tasks"
    (external / "beauty_large10000_100neg_test_same_candidate").mkdir(parents=True)
    (external / "books_large10000_100neg_test_same_candidate").mkdir(parents=True)

    plan = build_four_domain_server_plan(
        external_root=external,
        domains=["beauty", "books"],
        splits=["test"],
    )

    commands = plan["commands"]
    assert "build_week8_lora_sft" in commands
    assert "merge_week8_lora_sft" in commands
    assert "train_week8_lora_controls" in commands
    assert "evaluate_week8_lora_controls" in commands
    assert "merge_lora_sft_data.py" in commands["merge_week8_lora_sft"][0]
    assert "four_domain/history_only_sft" in commands["merge_week8_lora_sft"][0]
    assert "--datasets beauty books" in commands["merge_week8_lora_sft"][0]
    assert commands["train_week8_lora_controls"][0].startswith("mkdir -p ")
    assert "week8_lora_8b_rerank_eval.yaml" in commands["evaluate_week8_lora_controls"][0]
