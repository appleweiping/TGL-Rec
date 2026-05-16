from __future__ import annotations

from pathlib import Path

from llm4rec.experiments.four_domain_plan import (
    build_four_domain_server_plan,
    write_shell_runbook,
)


def test_four_domain_plan_detects_existing_and_missing_task_dirs(
    tmp_path: Path,
) -> None:
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
    (external / "beauty_supplementary_smallerN_100neg_valid_same_candidate").mkdir(
        parents=True
    )
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


def test_four_domain_plan_includes_week8_sft_merge_train_and_eval(
    tmp_path: Path,
) -> None:
    external = tmp_path / "external_tasks"
    (external / "beauty_supplementary_smallerN_100neg_test_same_candidate").mkdir(
        parents=True
    )
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
    assert (
        "week8_lora_8b_rerank_eval.yaml" in commands["evaluate_week8_lora_controls"][0]
    )


def test_four_domain_plan_includes_observation_and_pony_baseline_reuse_gates(
    tmp_path: Path,
) -> None:
    external = tmp_path / "external_tasks"
    (external / "beauty_supplementary_smallerN_100neg_test_same_candidate").mkdir(
        parents=True
    )

    plan = build_four_domain_server_plan(
        external_root=external,
        domains=["beauty"],
        splits=["test"],
    )

    run_order = plan["run_order"]
    commands = plan["commands"]
    required_sections = {
        "observation_qwen3_base",
        "pony_official_baseline_reuse",
        "pony_official_pending_baselines",
        "ours_framework_ablation_matrix",
    }

    assert required_sections.issubset(set(run_order))
    assert required_sections.issubset(set(commands))
    assert set(plan["pony_official_baseline_methods"]) >= {
        "llm2rec",
        "llmesr",
        "llmemb",
        "rlmrec",
    }
    assert plan["pony_official_pending_methods"] == ["promax"]
    assert (
        "week8_qwen3_8b_base_observation.yaml" in commands["observation_qwen3_base"][1]
    )
    assert "--limit 20" in commands["observation_qwen3_base"][1]
    assert plan["score_import_contract"]["required_importer"] == (
        "main_import_same_candidate_baseline_scores.py"
    )
    assert plan["score_import_contract"]["required_score_schema"] == (
        "source_event_id,user_id,item_id,score"
    )
    reuse_commands = " ".join(commands["pony_official_baseline_reuse"])
    assert "PONY_REUSE" in reuse_commands
    assert "configs/baselines/pony_official_external.yaml" in reuse_commands
    pending_commands = " ".join(commands["pony_official_pending_baselines"])
    assert "PENDING Pony official baseline" in pending_commands
    assert "promax" in pending_commands
    assert "nohup" not in reuse_commands
    assert "train_lora_8b.py" not in reuse_commands
    assert "CUDA_VISIBLE_DEVICES" not in reuse_commands
    ablation_commands = " ".join(commands["ours_framework_ablation_matrix"])
    assert "protocol_week8_large10000_same_candidate" in ablation_commands
    assert "no_need_gate" in ablation_commands


def test_four_domain_plan_uses_current_external_task_names_by_default(
    tmp_path: Path,
) -> None:
    external = tmp_path / "external_tasks"
    (external / "beauty_supplementary_smallerN_100neg_test_same_candidate").mkdir(
        parents=True
    )
    (external / "books_large10000_100neg_test_same_candidate").mkdir(parents=True)
    (external / "electronics_large10000_100neg_test_same_candidate").mkdir(parents=True)
    (external / "movies_large10000_100neg_test_same_candidate").mkdir(parents=True)

    plan = build_four_domain_server_plan(
        external_root=external,
        splits=["test"],
    )

    assert plan["missing_task_dirs"] == []
    import_command = plan["commands"]["import_frozen_same_candidate_tasks"][0]
    assert "beauty_supplementary_smallerN_100neg_test_same_candidate" in import_command
    assert "books_large10000_100neg_test_same_candidate" in import_command
    assert "electronics_large10000_100neg_test_same_candidate" in import_command
    assert "movies_large10000_100neg_test_same_candidate" in import_command


def test_four_domain_plan_allows_task_prefix_override(tmp_path: Path) -> None:
    external = tmp_path / "external_tasks"
    (external / "beauty_large10000_100neg_test_same_candidate").mkdir(parents=True)

    plan = build_four_domain_server_plan(
        external_root=external,
        domains=["beauty"],
        splits=["test"],
        task_prefixes={"beauty": "beauty_large10000_100neg"},
    )

    assert plan["missing_task_dirs"] == []
    assert "beauty_large10000_100neg_test_same_candidate" in (
        plan["commands"]["import_frozen_same_candidate_tasks"][0]
    )
