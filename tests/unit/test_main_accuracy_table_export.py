from pathlib import Path

from llm4rec.evaluation.main_table import export_main_accuracy_multiseed_tables
from llm4rec.io.artifacts import write_csv_rows


def test_main_accuracy_table_export_formats_mean_std_and_marker(tmp_path: Path):
    aggregate_rows = [
        {"dataset": "tiny", "mean": 0.5, "method": "time_graph_evidence", "metric": "Recall@5", "std": 0.1},
        {"dataset": "tiny", "mean": 0.4, "method": "time_graph_evidence", "metric": "NDCG@5", "std": 0.2},
        {"dataset": "tiny", "mean": 0.3, "method": "time_graph_evidence", "metric": "MRR@10", "std": 0.0},
        {"dataset": "tiny", "mean": 0.2, "method": "time_graph_evidence", "metric": "coverage", "std": 0.0},
        {"dataset": "tiny", "mean": 1.2, "method": "time_graph_evidence", "metric": "novelty", "std": 0.1},
        {"dataset": "tiny", "mean": 0.1, "method": "time_graph_evidence", "metric": "long_tail_ratio", "std": 0.0},
        {"dataset": "tiny", "mean": 9.0, "method": "time_graph_evidence", "metric": "runtime_seconds", "std": 1.0},
    ]
    significance_rows = [
        {
            "dataset": "tiny",
            "effect_direction": "method_a_better",
            "method_a": "time_graph_evidence",
            "method_b": "bm25",
            "metric": "Recall@5",
            "notes": "best_non_ours",
            "significant_at_0_05": "True",
        }
    ]
    write_csv_rows(tmp_path / "aggregate_metrics.csv", aggregate_rows)
    write_csv_rows(tmp_path / "significance_tests.csv", significance_rows)

    result = export_main_accuracy_multiseed_tables(tmp_path)

    assert Path(result["table_csv"]).is_file()
    text = Path(result["table_csv"]).read_text(encoding="utf-8")
    assert "0.500000±0.100000" in text
    assert ",*" in text
    assert Path(result["table_tex"]).is_file()
