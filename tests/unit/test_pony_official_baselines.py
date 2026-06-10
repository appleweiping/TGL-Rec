from __future__ import annotations

from pathlib import Path

import yaml

from llm4rec.baselines.pony_official import (
    pony_official_baseline_names,
    validate_pony_official_manifest,
)


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "configs" / "baselines" / "pony_official_external.yaml"


def test_pony_official_manifest_records_active_baseline_suite() -> None:
    result = validate_pony_official_manifest(MANIFEST)

    assert result["status"] == "pass"
    assert result["completed"] == [
        "llm2rec",
        "llmesr",
        "llmemb",
        "rlmrec",
        "irllrec",
        "elmrec",
        "proex",
        "promax",
    ]
    assert result["pending"] == []
    assert result["blocked"] == ["setrec"]
    assert result["baseline_count"] == 9


def test_pony_official_manifest_uses_same_candidate_score_contract() -> None:
    data = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))

    assert data["score_file_contract"]["required_schema"] == [
        "source_event_id",
        "user_id",
        "item_id",
        "score",
    ]
    assert data["score_file_contract"]["required_key_set"] == "exact_match_to_candidate_rows"
    assert data["score_file_contract"]["importer"] == "main_import_same_candidate_baseline_scores.py"
    assert data["baseline_artifact_policy"]["copy_large_archives_into_tglrec_git"] is False


def test_pony_official_baselines_have_provenance_fields() -> None:
    data = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))
    names = pony_official_baseline_names(MANIFEST)

    assert names == [
        "llm2rec",
        "llmesr",
        "llmemb",
        "rlmrec",
        "irllrec",
        "elmrec",
        "proex",
        "promax",
        "setrec",
    ]
    for name in names:
        spec = data["official_baselines"][name]
        assert spec["method_id"]
        assert spec["status"]
        assert spec["official_repo"].startswith("https://github.com/")
        assert spec["pinned_commit"]
        assert spec.get("summary_csv") or spec.get("evidence_archives") or spec.get("blocker")
