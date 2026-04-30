from pathlib import Path

import pandas as pd

from tglrec.data import schema
from tglrec.data.movielens import preprocess_movielens_1m
from tglrec.data.splits import apply_stable_ids, assert_no_future_leakage, training_events_as_of
from tglrec.utils.io import read_json


def _write_synthetic_ml1m(root: Path) -> Path:
    raw = root / "ml-1m"
    raw.mkdir(parents=True)
    ratings = [
        "2::20::5::100",
        "2::10::4::100",
        "2::30::4::120",
        "1::10::3::130",
        "1::20::4::140",
        "1::30::5::150",
        "3::30::4::160",
        "3::20::3::170",
        "3::10::5::180",
        "4::10::5::190",
        "4::20::4::200",
        "4::30::4::210",
    ]
    movies = [
        "10::Movie A (2000)::Drama",
        "20::Movie B (2000)::Comedy",
        "30::Movie C (2000)::Action",
    ]
    (raw / "ratings.dat").write_text("\n".join(ratings) + "\n", encoding="latin-1")
    (raw / "movies.dat").write_text("\n".join(movies) + "\n", encoding="latin-1")
    return raw


def test_movielens_preprocessing_writes_temporal_splits(tmp_path: Path):
    raw = _write_synthetic_ml1m(tmp_path / "raw")
    output = tmp_path / "processed"

    result = preprocess_movielens_1m(
        raw_dir=raw,
        output_dir=output,
        min_user_interactions=3,
        min_item_interactions=3,
        global_train_ratio=0.5,
        global_val_ratio=0.25,
        seed=2026,
    )

    interactions = pd.read_csv(output / "interactions.csv")
    assert result.num_interactions == 12
    assert (output / "config.yaml").exists()
    assert (output / "metadata.json").exists()
    assert (output / "checksums.json").exists()
    assert_no_future_leakage(interactions, schema.SPLIT_LOO)
    assert_no_future_leakage(interactions, schema.SPLIT_GLOBAL)
    metadata = read_json(output / "metadata.json")
    checksum_manifest = read_json(output / "checksums.json")
    assert metadata["same_user_timestamp_tie_stats"] == {
        "affected_users": 1,
        "max_events_at_same_timestamp": 2,
        "tied_extra_rows": 1,
        "tied_groups": 1,
        "tied_rows": 2,
    }
    assert metadata["processed_file_checksums"]["interactions.csv"]["sha256"]
    assert checksum_manifest["files"]["metadata.json"]["sha256"]
    assert checksum_manifest["files"]["temporal_leave_one_out/test.csv"]["bytes"] > 0

    per_user_counts = interactions.groupby(schema.USER_ID)[schema.SPLIT_LOO].value_counts()
    for user_id in interactions[schema.USER_ID].unique():
        assert per_user_counts[(user_id, "val")] == 1
        assert per_user_counts[(user_id, "test")] == 1

    assert pd.read_csv(output / "temporal_leave_one_out" / "train.csv")[schema.TIMESTAMP].max() < 200
    assert set(pd.read_csv(output / "items.csv")["title"]) == {
        "Movie A (2000)",
        "Movie B (2000)",
        "Movie C (2000)",
    }


def test_stable_id_mapping_independent_of_row_order():
    interactions = pd.DataFrame(
        {
            schema.RAW_USER_ID: ["u2", "u1", "u10", "u1"],
            schema.RAW_ITEM_ID: ["i2", "i1", "i10", "i2"],
            schema.TIMESTAMP: [4, 1, 3, 2],
            schema.RATING: [1, 1, 1, 1],
        }
    )

    first, users_first, items_first = apply_stable_ids(interactions)
    second, users_second, items_second = apply_stable_ids(
        interactions.sample(frac=1.0, random_state=7).reset_index(drop=True)
    )

    assert users_first.to_dict("records") == users_second.to_dict("records")
    assert items_first.to_dict("records") == items_second.to_dict("records")
    assert set(first[schema.USER_ID]) == set(second[schema.USER_ID])


def test_as_of_training_filter_excludes_cross_user_future_train_events(tmp_path: Path):
    raw = _write_synthetic_ml1m(tmp_path / "raw")
    output = tmp_path / "processed"
    preprocess_movielens_1m(
        raw_dir=raw,
        output_dir=output,
        min_user_interactions=3,
        min_item_interactions=3,
        global_train_ratio=0.5,
        global_val_ratio=0.25,
        seed=2026,
    )
    interactions = pd.read_csv(output / "interactions.csv")

    available = training_events_as_of(
        interactions,
        split_col=schema.SPLIT_LOO,
        prediction_timestamp=150,
    )

    assert available[schema.TIMESTAMP].max() < 150
    assert 190 not in set(available[schema.TIMESTAMP])
