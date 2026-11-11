"""Tratamento e preparação do German Credit para modelagem e monitoramento.

Decisões de tratamento (baseadas na análise exploratória):

- **Nulos / duplicados:** o dataset original não possui nenhum dos dois. Ainda
  assim, `clean` verifica ambos para proteger lotes futuros de produção.
- **Outliers:** mantidos. Valores extremos de `credit_amount`, `duration_months`
  e `age` são legítimos (não são erros de digitação) e removê-los enviesaria o
  modelo justamente na cauda que o drift econômico tende a deslocar.
- **Categorias raras:** não são agrupadas aqui; o agrupamento é feito no
  pipeline do modelo (`OneHotEncoder(min_frequency=...)`), para que a regra seja
  aprendida apenas no treino e aplicada igualmente em produção.
- **Feature derivada:** `monthly_installment` (valor / prazo) aproxima o peso
  da parcela - variável diretamente afetada por inflação.
- **Atributos sensíveis:** separados das features (LGPD art. 6º, IX) e mantidos
  apenas para auditoria de viés.

Uso:
    uv run python -m credit_scoring.data.preprocess
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import pandas as pd
from sklearn.model_selection import train_test_split

from credit_scoring.config import load_config
from credit_scoring.contracts.schemas import TRAINING_CONTRACT, validate
from credit_scoring.data.load import ID_COLUMN, SENSITIVE_COLUMNS, TARGET, load_german_credit
from credit_scoring.observability.logs import setup_logging
from credit_scoring.observability.metrics import track_stage

logger = logging.getLogger(__name__)

__all__ = ["ID_COLUMN", "FEATURES", "NUMERIC_FEATURES", "CATEGORICAL_FEATURES"]

NUMERIC_FEATURES: Final[list[str]] = [
    "duration_months",
    "credit_amount",
    "installment_rate",
    "residence_since",
    "existing_credits",
    "num_dependents",
    "monthly_installment",
]

CATEGORICAL_FEATURES: Final[list[str]] = [
    "checking_status",
    "credit_history",
    "purpose",
    "savings_status",
    "employment_since",
    "other_debtors",
    "property",
    "other_installment_plans",
    "housing",
    "job",
    "own_telephone",
]

FEATURES: Final[list[str]] = NUMERIC_FEATURES + CATEGORICAL_FEATURES


@dataclass(frozen=True)
class DatasetSplit:
    """Partições do dataset tratado.

    Attributes:
        train: Dados de treino (também usados como referência de drift).
        test: Dados de teste, estratificados pelo alvo.
    """

    train: pd.DataFrame
    test: pd.DataFrame


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica validações e correções básicas de qualidade.

    Remove linhas duplicadas e falha se houver valores nulos, já que o
    German Credit não possui ausentes por definição - nulos indicariam
    categoria desconhecida na decodificação ou falha na ingestão.

    Args:
        df: DataFrame decodificado (saída de `load_german_credit`).

    Inteiros são mantidos em `int64`: reduzir para `int16` (máx. 32.767)
    estouraria silenciosamente `credit_amount` em cenários de inflação.

    Returns:
        Novo DataFrame sem duplicatas.

    Raises:
        ValueError: Se houver valores nulos em qualquer coluna.
    """
    nulls = df.isna().sum()
    if nulls.any():
        raise ValueError(f"Valores nulos encontrados: {nulls[nulls > 0].to_dict()}")

    n_dup = int(df.duplicated().sum())
    if n_dup:
        logger.warning("Removendo %d linhas duplicadas", n_dup)
    return df.drop_duplicates().reset_index(drop=True)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cria features derivadas.

    Args:
        df: DataFrame tratado por `clean`.

    Returns:
        Novo DataFrame com a coluna `monthly_installment` (valor do crédito
        dividido pelo prazo em meses, arredondado em 2 casas).
    """
    out = df.copy()
    out["monthly_installment"] = (out["credit_amount"] / out["duration_months"]).round(2)
    return out


def add_id(df: pd.DataFrame) -> pd.DataFrame:
    """Adiciona um identificador pseudonimizado para cada solicitação.

    O identificador é sequencial e não deriva de nenhum dado pessoal, permitindo
    rastrear predições e rótulos tardios (necessário para detectar concept
    drift) sem expor o titular.

    Args:
        df: DataFrame de entrada.

    Returns:
        Novo DataFrame com `application_id` como primeira coluna.
    """
    out = df.copy()
    out.insert(0, ID_COLUMN, [f"APP-{i:05d}" for i in range(len(out))])
    return out


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Separa features do modelo, alvo e atributos sensíveis.

    Args:
        df: DataFrame tratado contendo features, alvo e colunas sensíveis.

    Returns:
        Tupla `(X, y, sensitive)`:
            - `X`: apenas as colunas de `FEATURES`.
            - `y`: série do alvo `default`.
            - `sensitive`: colunas de `SENSITIVE_COLUMNS`, para auditoria de viés.
    """
    return df[FEATURES], df[TARGET], df[SENSITIVE_COLUMNS]


def build_dataset() -> pd.DataFrame:
    """Executa o tratamento completo a partir do arquivo bruto.

    O resultado é validado contra `TRAINING_CONTRACT` antes de ser retornado:
    nenhum dado fora do contrato chega ao treino.

    Returns:
        DataFrame com `application_id`, features, colunas sensíveis e alvo.

    Raises:
        pandera.errors.SchemaErrors: Se o dataset violar o contrato de dados.
    """
    df = load_german_credit()
    df = clean(df)
    df = add_features(df)
    df = add_id(df)
    return validate(df[[ID_COLUMN, *FEATURES, *SENSITIVE_COLUMNS, TARGET]], TRAINING_CONTRACT)


def train_test(df: pd.DataFrame, test_size: float = 0.2, seed: int = 42) -> DatasetSplit:
    """Divide o dataset em treino e teste, estratificando pelo alvo.

    Args:
        df: DataFrame tratado (saída de `build_dataset`).
        test_size: Proporção destinada ao teste.
        seed: Semente para reprodutibilidade.

    Returns:
        `DatasetSplit` com as partições de treino e teste.
    """
    train, test = train_test_split(
        df, test_size=test_size, stratify=df[TARGET], random_state=seed
    )
    return DatasetSplit(train=train.reset_index(drop=True), test=test.reset_index(drop=True))


def save_split(split: DatasetSplit, processed_dir: Path | str, reference_dir: Path | str) -> None:
    """Persiste as partições em Parquet (preserva os dtypes categóricos).

    O treino também é salvo como `reference.parquet`: é a distribuição base
    contra a qual os lotes de produção serão comparados na detecção de drift.

    Args:
        split: Partições de treino e teste.
        processed_dir: Diretório de saída de `train.parquet` e `test.parquet`.
        reference_dir: Diretório de saída de `reference.parquet`.
    """
    processed_dir, reference_dir = Path(processed_dir), Path(reference_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    reference_dir.mkdir(parents=True, exist_ok=True)

    split.train.to_parquet(processed_dir / "train.parquet", index=False)
    split.test.to_parquet(processed_dir / "test.parquet", index=False)
    split.train.to_parquet(reference_dir / "reference.parquet", index=False)


def main() -> None:
    """Gera os datasets tratados conforme `configs/config.yaml`."""
    cfg = load_config()
    with track_stage("preprocess") as metrics:
        df = build_dataset()
        split = train_test(df, test_size=cfg["model"]["test_size"], seed=cfg["seed"])
        save_split(split, cfg["paths"]["processed"], cfg["paths"]["reference"])
        for name, part in (("train", split.train), ("test", split.test)):
            metrics.set("dataset_rows", len(part), "Linhas por partição", split=name)
            metrics.set(
                "dataset_default_rate", float(part[TARGET].mean()),
                "Taxa de default por partição", split=name,
            )
    logger.info(
        "Treino: %d linhas (default %.1f%%) | Teste: %d linhas (default %.1f%%)",
        len(split.train),
        100 * split.train[TARGET].mean(),
        len(split.test),
        100 * split.test[TARGET].mean(),
    )


if __name__ == "__main__":
    setup_logging()
    main()
