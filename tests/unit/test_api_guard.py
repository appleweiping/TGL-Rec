from pathlib import Path

import pytest

from llm4rec.llm.api_guard import APIGuardConfig, APIGuardError, validate_api_guard


def _guard(tmp_path: Path, **overrides):
    values = {
        "run_mode": "diagnostic_api",
        "allow_api_calls": True,
        "provider": "openai_compatible",
        "api_key_env": "PHASE3B_TEST_API_KEY",
        "max_api_calls": 125,
        "estimated_calls": 10,
        "cache_policy": "read_write",
        "run_dir": tmp_path / "run",
        "resume": False,
    }
    values.update(overrides)
    return APIGuardConfig(**values)


def test_api_guard_allows_safe_real_api_config(tmp_path, monkeypatch):
    monkeypatch.setenv("PHASE3B_TEST_API_KEY", "secret")
    validate_api_guard(_guard(tmp_path))


def test_api_guard_blocks_allow_api_false(tmp_path, monkeypatch):
    monkeypatch.setenv("PHASE3B_TEST_API_KEY", "secret")
    with pytest.raises(APIGuardError, match="allow_api_calls"):
        validate_api_guard(_guard(tmp_path, allow_api_calls=False))


def test_api_guard_blocks_wrong_run_mode(tmp_path, monkeypatch):
    monkeypatch.setenv("PHASE3B_TEST_API_KEY", "secret")
    with pytest.raises(APIGuardError, match="run_mode"):
        validate_api_guard(_guard(tmp_path, run_mode="diagnostic_mock"))


def test_api_guard_blocks_missing_api_key(tmp_path, monkeypatch):
    monkeypatch.delenv("PHASE3B_TEST_API_KEY", raising=False)
    with pytest.raises(APIGuardError, match="Missing API key"):
        validate_api_guard(_guard(tmp_path))


def test_api_guard_blocks_call_cap_exceeded(tmp_path, monkeypatch):
    monkeypatch.setenv("PHASE3B_TEST_API_KEY", "secret")
    with pytest.raises(APIGuardError, match="estimated_calls exceeds"):
        validate_api_guard(_guard(tmp_path, estimated_calls=126, max_api_calls=125))


def test_api_guard_blocks_mock_provider_in_api_mode(tmp_path, monkeypatch):
    monkeypatch.setenv("PHASE3B_TEST_API_KEY", "secret")
    with pytest.raises(APIGuardError, match="provider=mock"):
        validate_api_guard(_guard(tmp_path, provider="mock"))
