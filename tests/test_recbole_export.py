import csv
import json
from pathlib import Path

import pandas as pd
import pytest

from tglrec.cli import main
from tglrec.data import schema
from tglrec.data.recbole_export import export_recbole_general_cf


def _write_processed_dataset(root: Path, *, write_users: bool = True) -> None:
    root.mkdir(parents=True)
    interactions = pd.DataFrame(
        [
            {
                schema.EVENT_ID: 0,
                schema.USER_ID: 0,
                schema.ITEM_ID: 10,
                schema.RAW_USER_ID: "u0",
                schema.RAW_ITEM_ID: "i10",
                schema.TIMESTAMP: 10,
                schema.RATING: 1.0,
                schema.SPLIT_LOO: "train",
                schema.SPLIT_GLOBAL: "train",
            },
            {
                schema.EVENT_ID: 1,
                schema.USER_ID: 0,
                schema.ITEM_ID: 11,
                schema.RAW_USER_ID: "u0",
                schema.RAW_ITEM_ID: "i11",
                schema.TIMESTAMP: 20,
                schema.RATING: 1.0,
                schema.SPLIT_LOO: "val",
                schema.SPLIT_GLOBAL: "val",
            },
            {
                schema.EVENT_ID: 2,
                schema.USER_ID: 0,
                schema.ITEM_ID: 12,
                schema.RAW_USER_ID: "u0",
                schema.RAW_ITEM_ID: "i12",
                schema.TIMESTAMP: 30,
                schema.RATING: 1.0,
                schema.SPLIT_LOO: "test",
                schema.SPLIT_GLOBAL: "test",
            },
            {
                schema.EVENT_ID: 3,
                schema.USER_ID: 1,
                schema.ITEM_ID: 10,
                schema.RAW_USER_ID: "u1",
                schema.RAW_ITEM_ID: "i10",
                schema.TIMESTAMP: 11,
                schema.RATING: 1.0,
                schema.SPLIT_LOO: "train",
                schema.SPLIT_GLOBAL: "train",
            },
            {
                schema.EVENT_ID: 4,
                schema.USER_ID: 1,
                schema.ITEM_ID: 11,
                schema.RAW_USER_ID: "u1",
                schema.RAW_ITEM_ID: "i11",
                schema.TIMESTAMP: 21,
                schema.RATING: 1.0,
                schema.SPLIT_LOO: "val",
                schema.SPLIT_GLOBAL: "val",
            },
            {
                schema.EVENT_ID: 5,
                schema.USER_ID: 1,
                schema.ITEM_ID: 12,
                schema.RAW_USER_ID: "u1",
                schema.RAW_ITEM_ID: "i12",
                schema.TIMESTAMP: 31,
                schema.RATING: 1.0,
                schema.SPLIT_LOO: "test",
                schema.SPLIT_GLOBAL: "test",
            },
        ]
    )
    interactions.to_csv(root / "interactions.csv", index=False)
    if write_users:
        pd.DataFrame(
            [
                {schema.USER_ID: 0, schema.RAW_USER_ID: "u0"},
                {schema.USER_ID: 1, schema.RAW_USER_ID: "u1"},
            ]
        ).to_csv(root / "users.csv", index=False)
    pd.DataFrame(
        [
            {schema.ITEM_ID: 10, schema.RAW_ITEM_ID: "i10", "title": "First Item"},
            {schema.ITEM_ID: 11, schema.RAW_ITEM_ID: "i11", "title": "Second\tItem"},
            {schema.ITEM_ID: 12, schema.RAW_ITEM_ID: "i12", "title": "Third Item"},
        ]
    ).to_csv(root / "items.csv", index=False)


def test_recbole_general_export_cli_writes_atomic_benchmark_files(tmp_path: Path, capsys):
    dataset = tmp_path / "processed"
    output = tmp_path / "recbole"
    _write_processed_dataset(dataset)

    assert (
        main(
            [
                "export",
                "recbole-general",
                "--dataset-dir",
                str(dataset),
                "--output-dir",
                str(output),
                "--dataset-name",
                "toy_loo",
            ]
        )
        == 0
    )
    captured = capsys.readouterr()
    assert "wrote RecBole general-CF export:" in captured.out
    data_root = output / "toy_loo"
    train_path = data_root / "toy_loo.train.inter"
    valid_path = data_root / "toy_loo.valid.inter"
    test_path = data_root / "toy_loo.test.inter"
    for path in [
        train_path,
        valid_path,
        test_path,
        data_root / "toy_loo.user",
        data_root / "toy_loo.item",
        output / "README.md",
        output / "recbole_general_cf.yaml",
        output / "metadata.json",
        output / "checksums.json",
    ]:
        assert path.exists()

    assert _read_tsv(train_path)[0] == [
        "user_id:token",
        "item_id:token",
        "rating:float",
        "timestamp:float",
        "event_id:token",
    ]
    assert len(_read_tsv(train_path)) == 3
    assert len(_read_tsv(valid_path)) == 3
    assert len(_read_tsv(test_path)) == 3
    config = (output / "recbole_general_cf.yaml").read_text(encoding="utf-8")
    assert "benchmark_filename:" in config
    assert "- train" in config
    assert "mode: full" in config
    metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["export_type"] == "recbole_general_cf_benchmark"
    assert metadata["num_train"] == 2
    assert "Sequential RecBole models need" in metadata["warning"]
    checksums = json.loads((output / "checksums.json").read_text(encoding="utf-8"))
    assert "toy_loo/toy_loo.train.inter" in checksums["files"]


def test_recbole_general_export_is_deterministic_and_rejects_missing_metadata(
    tmp_path: Path,
):
    dataset = tmp_path / "processed"
    _write_processed_dataset(dataset)
    first = tmp_path / "first"
    second = tmp_path / "second"

    result = export_recbole_general_cf(
        dataset_dir=dataset,
        output_dir=first,
        dataset_name="toy",
        command="synthetic",
    )
    export_recbole_general_cf(
        dataset_dir=dataset,
        output_dir=second,
        dataset_name="toy",
        command="synthetic",
    )

    assert result.num_train == 2
    assert (first / "toy" / "toy.train.inter").read_text(encoding="utf-8") == (
        second / "toy" / "toy.train.inter"
    ).read_text(encoding="utf-8")

    missing_users_dataset = tmp_path / "missing_users"
    _write_processed_dataset(missing_users_dataset, write_users=False)
    with pytest.raises(FileNotFoundError, match="Missing processed users"):
        export_recbole_general_cf(dataset_dir=missing_users_dataset, output_dir=tmp_path / "bad")


def _read_tsv(path: Path) -> list[list[str]]:
    with path.open(newline="") as handle:
        return list(csv.reader(handle, delimiter="\t"))
