from llm4rec.data.readiness import NO_EXECUTION_FLAG
from llm4rec.evaluation.table_plan import plan_paper_tables
from llm4rec.experiments.launch_manifest import create_launch_manifest


def test_table_plan_has_shells_without_numbers(tmp_path):
    manifest = create_launch_manifest(tmp_path / "launch_manifest.json")
    plan = plan_paper_tables(manifest, tmp_path / "table_plan.json")
    assert plan[NO_EXECUTION_FLAG] is True
    assert plan["numeric_values_present"] is False
    assert len(plan["tables"]) == 6
    assert all(table["values"] == [] for table in plan["tables"])
