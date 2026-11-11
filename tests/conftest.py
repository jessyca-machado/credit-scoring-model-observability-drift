"""Configuração global dos testes: isola a observabilidade do ambiente real."""

import pytest


@pytest.fixture(autouse=True)
def _isolate_observability(monkeypatch, tmp_path):
    """Desativa o envio ao Pushgateway e grava logs num diretório temporário."""
    monkeypatch.setenv("PUSHGATEWAY_URL", "")
    monkeypatch.setenv("LOG_DIR", str(tmp_path / "logs"))
