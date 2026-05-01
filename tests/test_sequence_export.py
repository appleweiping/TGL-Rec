import csv
import json
from pathlib import Path

import pandas as pd
import pytest

from tglrec.cli import main
from tglrec.data import schema
from tglrec.data.sequence_export import export_sequence_cases


def _write_processed_dataset(root: Path) -> None:
    root.mkdir(parents=True)
    rows = []
    event_id = 0
    for user_id, item_ids in [(0, [10, 11, 12, 13]), (1, [20, 21, 22, 23])]:
        for offset, item_id in enumerate(item_ids):
            split = "train" if offset < 2 else "val" if offset == 2 else "test"
            rows.append(
                {
                    schema.EVENT_ID: event_id,
                    schema.USER_ID: user_id,
                    schema.ITEM_ID: item_id,
                    schema.RAW_USER_ID: f"u{user_id}",
                    schema.RAW_ITEM_ID: f"i{item_id}",
                    schema.TIMESTAMP: 100 * user_id + offset + 1,
                    schema.RATING: 1.0,
                    schema.SPLIT_LOO: split,
                    schema.SPLIT_GLOBAL: split,
                }
            )
            event_id += 1
    pd.DataFrame(rows).to_csv(root / "interactions.csv", index=False)


def test_sequence_case_export_cli_writes_histories_without_future_leakage(tmp_path: Path, capsys):
    dataset = tmp_path / "processed"
    output = tmp_path / "sequence_cases"
    _write_processed_dataset(dataset)

    assert (
        main(
            [
                "export",
                "sequence-cases",
                "--dataset-dir",
                str(dataset),
                "--output-dir",
                str(output),
                "--dataset-name",
                "toy_seq",
            ]
        )
        == 0
    )
    captured = capsys.readouterr()
    assert "wrote sequence case export:" in captured.out
    for name in [
        "README.md",
        "checksums.json",
        "command.txt",
        "config.yaml",
        "eval_cases.csv",
        "git_commit.txt",
        "metadata.json",
        "train_examples.csv",
        "user_sequences.csv",
    ]:
        assert (output / name).exists()

    train_rows = _read_csv(output / "train_examples.csv")
    eval_rows = _read_csv(output / "eval_cases.csv")
    sequence_rows = _read_csv(output / "user_sequences.csv")
    assert train_rows == []
    assert json.loads(sequence_rows[0]["train_item_ids_json"]) == [10, 11]
    val_case = next(row for row in eval_rows if row["target_split"] == "val" and row["user_id"] == "0")
    test_case = next(row for row in eval_rows if row["target_split"] == "test" and row["user_id"] == "0")
    assert json.loads(val_case["history_item_ids_json"]) == [10, 11]
    assert json.loads(test_case["history_item_ids_json"]) == [10, 11, 12]
    assert 13 not in json.loads(test_case["history_item_ids_json"])
    metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["num_train_examples"] == 0
    assert metadata["num_train_transitions_available"] == 2
    assert metadata["num_validation_cases"] == 2
    assert metadata["num_test_cases"] == 2
    assert metadata["train_examples_materialized"] is False


def test_sequence_case_export_can_exclude_validation_history_and_rejects_bad_split(
    tmp_path: Path,
):
    dataset = tmp_path / "processed"
    output = tmp_path / "sequence_cases"
    _write_processed_dataset(dataset)

    result = export_sequence_cases(
        dataset_dir=dataset,
        output_dir=output,
        dataset_name="toy_seq",
        use_validation_history_for_test=False,
        write_train_examples=True,
        command="synthetic",
    )

    assert result.num_test_cases == 2
    assert result.num_train_examples == 2
    train_example = _read_csv(output / "train_examples.csv")[0]
    assert json.loads(train_example["history_item_ids_json"]) == [10]
    test_case = next(
        row
        for row in _read_csv(output / "eval_cases.csv")
        if row["target_split"] == "test" and row["user_id"] == "0"
    )
    assert json.loads(test_case["history_item_ids_json"]) == [10, 11]

    with pytest.raises(ValueError, match="split_name must be"):
        export_sequence_cases(dataset_dir=dataset, output_dir=tmp_path / "bad", split_name="random")
    with pytest.raises(ValueError, match="max_history_items"):
        export_sequence_cases(dataset_dir=dataset, output_dir=tmp_path / "bad_len", max_history_items=-1)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))
