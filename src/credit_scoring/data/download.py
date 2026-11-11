"""Download do German Credit Data (Statlog) do UCI Machine Learning Repository.

Fonte: https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data
Licença: CC BY 4.0 - Hofmann, H. (1994).

Uso:
    uv run python -m credit_scoring.data.download
"""

from __future__ import annotations

import hashlib
import logging
import urllib.request
from pathlib import Path
from typing import Final

from credit_scoring.observability.logs import setup_logging
from credit_scoring.observability.metrics import track_stage

logger = logging.getLogger(__name__)

URL: Final[str] = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"
SHA256: Final[str] = "b21f3d81db8071257d5ff1deaeba1fd4303b62712e6fcc9715c7a86202cb5871"
DEFAULT_PATH: Final[Path] = Path("data/raw/german.data")


def _sha256(path: Path) -> str:
    """Calcula o hash SHA-256 de um arquivo.

    Args:
        path: Caminho do arquivo.

    Returns:
        Hash SHA-256 em hexadecimal.
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()


def download(dest: Path | str = DEFAULT_PATH, force: bool = False) -> Path:
    """Baixa o German Credit Data e valida sua integridade.

    Se o arquivo já existir em `dest` com o checksum esperado, o download é
    pulado. O conteúdo é baixado para um arquivo temporário e só substitui o
    destino após a validação do SHA-256, evitando arquivos corrompidos.

    Args:
        dest: Caminho de destino do arquivo. Padrão: `data/raw/german.data`.
        force: Se True, baixa novamente mesmo que o arquivo já exista íntegro.

    Returns:
        Caminho do arquivo baixado.

    Raises:
        ValueError: Se o checksum do arquivo baixado não corresponder a `SHA256`.
        urllib.error.URLError: Se houver falha de rede ao acessar a UCI.

    Example:
        >>> path = download()
        >>> path.name
        'german.data'
    """
    dest = Path(dest)
    if dest.exists() and not force and _sha256(dest) == SHA256:
        logger.info("Dataset já presente e íntegro em %s", dest)
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".tmp")
    logger.info("Baixando %s", URL)
    urllib.request.urlretrieve(URL, tmp)

    digest = _sha256(tmp)
    if digest != SHA256:
        tmp.unlink(missing_ok=True)
        raise ValueError(f"Checksum inválido: esperado {SHA256}, obtido {digest}")

    tmp.replace(dest)
    logger.info("Dataset salvo em %s", dest)
    return dest


if __name__ == "__main__":
    setup_logging()
    with track_stage("download") as metrics:
        path = download()
        metrics.set("dataset_raw_bytes", path.stat().st_size, "Tamanho do arquivo bruto baixado")
