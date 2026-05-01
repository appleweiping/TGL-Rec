from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_phase2a_diagnostics_write_required_artifacts() -> None:
    root = Path(__file__).resolve().parents[2]
    subprocess.run(
        [
            sys.executable,
            str(root / "scripts" / "run_diagnostics.py"),
            "--config",
            str(root / "configs" / "diagnostics" / "similarity_vs_transition.yaml"),
        ],
        cwd=root,
        check=True,
    )
    diagnostics_dir = root / "outputs" / "runs" / "phase2a_diagnostics" / "diagnostics"
    expected = [
        "sequence_perturbation.json",
        "time_features.jsonl",
        "transition_edges.jsonl",
        "time_window_edges.jsonl",
        "similarity_vs_transition.json",
        "diagnostics_summary.json",
    ]
    for name in expected:
        assert (diagnostics_dir / name).is_file()
    summary = json.loads((diagnostics_dir / "diagnostics_summary.json").read_text(encoding="utf-8"))
    assert summary["transition_edge_count"] > 0
    assert summary["time_window_edge_count"] > 0
