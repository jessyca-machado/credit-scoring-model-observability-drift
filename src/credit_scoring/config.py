"""Leitura das configurações do projeto (`configs/config.yaml`)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Final

import yaml

CONFIG_PATH: Final[Path] = Path("configs/config.yaml")


def load_config(path: Path | str = CONFIG_PATH) -> dict[str, Any]:
    """Carrega o arquivo de configuração YAML.

    Args:
        path: Caminho do arquivo. Padrão: `configs/config.yaml`.

    Returns:
        Dicionário com as configurações.

    Raises:
        FileNotFoundError: Se o arquivo não existir.
    """
    with Path(path).open(encoding="utf-8") as f:
        return yaml.safe_load(f)
