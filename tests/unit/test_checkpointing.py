import pytest

from llm4rec.trainers.checkpointing import CheckpointError, load_checkpoint, save_checkpoint


def test_checkpoint_round_trip(tmp_path):
    path = tmp_path / "checkpoint.json"
    save_checkpoint(path, {"method": "mf", "seed": 1})
    assert load_checkpoint(path)["method"] == "mf"
    with pytest.raises(CheckpointError):
        save_checkpoint(path, {"method": "mf"}, overwrite=False)
