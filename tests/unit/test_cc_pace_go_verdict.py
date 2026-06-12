"""Smoke tests for the CC-PACE beauty GO/KILL verdict script."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "cc_pace_go_verdict.py"
SPEC = importlib.util.spec_from_file_location("cc_pace_go_verdict", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
go_verdict = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(go_verdict)


def _write_variant(dir_: Path, name: str, ndcg10: float, rows: list[float]) -> None:
    summary = {
        "variant": name,
        "n_examples": len(rows),
        "mock": False,
        "metrics": {"NDCG@10": ndcg10},
    }
    (dir_ / f"{name}.json").write_text(json.dumps(summary), encoding="utf-8")
    with (dir_ / f"{name}.json.per_user.jsonl").open("w", encoding="utf-8") as fh:
        for i, val in enumerate(rows):
            fh.write(json.dumps({"user_id": f"u{i}", "NDCG@10": val}) + "\n")


def test_go_verdict_writes_go_when_full_passes_gap(tmp_path, monkeypatch):
    _write_variant(tmp_path, "full", 0.131, [1.0, 0.5, 0.0])
    _write_variant(tmp_path, "text_only", 0.111, [0.0, 0.5, 0.0])
    out = tmp_path / "go_verdict.json"

    monkeypatch.setattr(
        "sys.argv", ["cc_pace_go_verdict.py", "--dir", str(tmp_path), "--out", str(out)]
    )
    go_verdict.main()

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["decision"] == "GO"
    assert payload["criteria"]["full_ndcg10_ge_0.13"]
    assert payload["criteria"]["full_gt_text_only"]
    assert payload["cf_gap"]["n_paired"] == 3


def test_go_verdict_writes_kill_when_full_misses_threshold(tmp_path, monkeypatch):
    _write_variant(tmp_path, "full", 0.129, [1.0, 0.0])
    _write_variant(tmp_path, "text_only", 0.1108, [0.0, 0.0])
    out = tmp_path / "go_verdict.json"

    monkeypatch.setattr(
        "sys.argv", ["cc_pace_go_verdict.py", "--dir", str(tmp_path), "--out", str(out)]
    )
    go_verdict.main()

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["decision"] == "KILL_OR_REFRAME"
    assert not payload["criteria"]["full_ndcg10_ge_0.13"]

