import pytest

from tglrec.cli import main


def test_cli_version(capsys):
    assert main(["--version"]) == 0
    captured = capsys.readouterr()
    assert captured.out.strip()


def test_cli_check_config(tmp_path, capsys):
    config = tmp_path / "config.yaml"
    config.write_text("seed: 2026\n", encoding="utf-8")

    assert main(["check-config", str(config)]) == 0
    captured = capsys.readouterr()
    assert "top-level keys=['seed']" in captured.out


def test_cli_missing_nested_subcommand_exits_nonzero():
    with pytest.raises(SystemExit) as exc_info:
        main(["preprocess"])

    assert exc_info.value.code != 0
