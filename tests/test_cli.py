"""CLI testleri (doctor / status / arguman analizi)."""

from __future__ import annotations

from pathlib import Path

import yaml

from main import build_parser, main


def test_parser_defaults() -> None:
    parser = build_parser()
    args = parser.parse_args([])
    assert args.command is None
    assert args.config is None

    run_args = parser.parse_args(["run", "--cycles", "2", "--backend", "langgraph"])
    assert run_args.command == "run"
    assert run_args.cycles == 2
    assert run_args.backend == "langgraph"


def test_doctor_returns_zero(config_file: Path, capsys) -> None:
    code = main(["--config", str(config_file), "doctor"])
    captured = capsys.readouterr().out
    assert code == 0
    assert "ortam kontrolu" in captured
    assert "paket:pandas" in captured


def test_status_without_database_returns_one(settings, tmp_path: Path, capsys) -> None:
    payload = settings.as_dict(relative=False)
    payload["memory"]["database"] = str(tmp_path / "missing.db")
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    code = main(["--config", str(path), "status"])
    assert code == 1
    assert "Veritabani bulunamadi" in capsys.readouterr().out


def test_invalid_config_returns_two(tmp_path: Path, capsys) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("exchange:\n  timeframe: 9x\n", encoding="utf-8")
    code = main(["--config", str(path), "doctor"])
    assert code == 2
    assert "Konfigurasyon hatasi" in capsys.readouterr().err
