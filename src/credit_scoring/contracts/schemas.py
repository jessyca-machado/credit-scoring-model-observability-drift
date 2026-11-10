"""Contratos de qualidade de dados do Credit Scoring (Pandera).

O contrato define o que é um registro **válido**, não o que é um registro
**típico**. As regras refletem o dicionário de dados da UCI e regras de
negócio/legais (ex.: maioridade), e não os valores observados no treino:
uma mudança de distribuição causada pela economia (ex.: valores de crédito
maiores por inflação) é *drift* e deve ser detectada pelo monitoramento, não
rejeitada pelo contrato. Já um valor impossível (idade 15, parcela negativa,
categoria inexistente) é *erro de dado* e deve barrar o pipeline.

Dois contratos são disponibilizados:

- `INFERENCE_CONTRACT`: lotes de produção. O alvo é opcional, pois o rótulo
  (pagou / não pagou) só chega meses depois da concessão.
- `TRAINING_CONTRACT`: dados de treino/teste. O alvo é obrigatório.
"""

from __future__ import annotations

import logging
from typing import Final

import pandas as pd
import pandera.pandas as pa
from pandera.errors import SchemaErrors

from credit_scoring.data.load import (
    CATEGORY_MAPS,
    ID_COLUMN,
    PERSONAL_STATUS_MAP,
    TARGET,
    category_dtype,
)

logger = logging.getLogger(__name__)

MIN_AGE: Final[int] = 18
MAX_AGE: Final[int] = 100
MAX_DURATION_MONTHS: Final[int] = 120
MAX_CREDIT_AMOUNT: Final[int] = 1_000_000
INSTALLMENT_TOLERANCE: Final[float] = 0.01

VALID_PERSONAL_STATUS: Final[set[tuple[str, str]]] = set(PERSONAL_STATUS_MAP.values())


def _categorical(column: str, description: str) -> pa.Column:
    """Cria a regra de uma coluna categórica com domínio fechado.

    Args:
        column: Nome da coluna.
        description: Descrição de negócio da coluna.

    Returns:
        Coluna Pandera que exige o dtype categórico do treino e valores
        pertencentes ao dicionário de dados.
    """
    dtype = category_dtype(column)
    return pa.Column(
        dtype,
        pa.Check.isin(list(dtype.categories)),
        nullable=False,
        description=description,
    )


def _no_duplicate_records(df: pd.DataFrame) -> pd.Series:
    """Marca como inválidas as solicitações repetidas (ignorando o ID).

    Args:
        df: Lote a validar.

    Returns:
        Série booleana; False nas linhas que repetem uma anterior.
    """
    return ~df.drop(columns=[ID_COLUMN, TARGET], errors="ignore").duplicated()


def _installment_is_consistent(df: pd.DataFrame) -> pd.Series:
    """Verifica se a parcela mensal corresponde a valor / prazo.

    Args:
        df: Lote a validar.

    Returns:
        Série booleana; False nas linhas com parcela inconsistente.
    """
    expected = df["credit_amount"] / df["duration_months"]
    return (df["monthly_installment"] - expected).abs() <= INSTALLMENT_TOLERANCE


def _personal_status_is_valid(df: pd.DataFrame) -> pd.Series:
    """Verifica se a combinação sexo + estado civil existe no dicionário UCI.

    Args:
        df: Lote a validar.

    Returns:
        Série booleana; False nas combinações inexistentes
        (ex.: `female` + `married/widowed`).
    """
    pairs = zip(df["sex"].astype(str), df["marital_status"].astype(str), strict=True)
    return pd.Series([p in VALID_PERSONAL_STATUS for p in pairs], index=df.index)


def build_contract(require_target: bool) -> pa.DataFrameSchema:
    """Monta o contrato de dados do Credit Scoring.

    Args:
        require_target: Se True, a coluna `default` é obrigatória (treino).
            Se False, é opcional (produção, rótulo tardio).

    Returns:
        `DataFrameSchema` estrito: colunas não previstas são rejeitadas.
    """
    columns: dict[str, pa.Column] = {
        ID_COLUMN: pa.Column(
            str,
            pa.Check.str_matches(r"^APP-\d{5,}$"),
            unique=True,
            description="Identificador pseudonimizado da solicitação",
        ),
        # Numéricas
        "duration_months": pa.Column(
            "int64",
            pa.Check.in_range(1, MAX_DURATION_MONTHS),
            description="Prazo do crédito em meses",
        ),
        "credit_amount": pa.Column(
            "int64",
            [pa.Check.gt(0), pa.Check.le(MAX_CREDIT_AMOUNT)],
            description="Valor solicitado (DM). Positivo; limite superior apenas de sanidade",
        ),
        "installment_rate": pa.Column(
            "int64",
            pa.Check.isin([1, 2, 3, 4]),
            description="Parcela em % da renda disponível (faixas codificadas 1-4)",
        ),
        "residence_since": pa.Column(
            "int64",
            pa.Check.isin([1, 2, 3, 4]),
            description="Tempo na residência atual (faixas codificadas 1-4)",
        ),
        "existing_credits": pa.Column(
            "int64",
            pa.Check.isin([1, 2, 3, 4]),
            description="Créditos existentes no banco (faixas codificadas 1-4)",
        ),
        "num_dependents": pa.Column(
            "int64",
            pa.Check.isin([1, 2]),
            description="Dependentes (1 = 0-2, 2 = 3 ou mais)",
        ),
        "monthly_installment": pa.Column(
            "float64",
            pa.Check.gt(0),
            description="Feature derivada: credit_amount / duration_months",
        ),
        # Sensíveis (auditoria de viés, não são features)
        "age": pa.Column(
            "int64",
            pa.Check.in_range(MIN_AGE, MAX_AGE),
            description="Idade em anos. Mínimo 18 (capacidade civil - CC art. 5º)",
        ),
        "sex": _categorical("sex", "Sexo (derivado de personal_status) - dado sensível"),
        "marital_status": _categorical("marital_status", "Estado civil - dado sensível"),
        TARGET: pa.Column(
            "int8",
            pa.Check.isin([0, 1]),
            required=require_target,
            description="Alvo: 1 = mau pagador, 0 = bom pagador",
        ),
    }
    for col in CATEGORY_MAPS:
        columns[col] = _categorical(col, f"Categórica do dicionário UCI: {col}")

    return pa.DataFrameSchema(
        columns,
        checks=[
            pa.Check(_no_duplicate_records, name="no_duplicate_records"),
            pa.Check(_installment_is_consistent, name="monthly_installment_consistent"),
            pa.Check(_personal_status_is_valid, name="valid_sex_marital_status_pair"),
        ],
        strict=True,
        name="credit_scoring_training" if require_target else "credit_scoring_inference",
    )


TRAINING_CONTRACT: Final[pa.DataFrameSchema] = build_contract(require_target=True)
INFERENCE_CONTRACT: Final[pa.DataFrameSchema] = build_contract(require_target=False)


def contract_report(df: pd.DataFrame, contract: pa.DataFrameSchema) -> pd.DataFrame:
    """Valida um lote sem interromper o fluxo e retorna as violações.

    Útil no monitoramento: o número de violações por regra pode ser exposto
    como métrica e o lote enviado para quarentena.

    Args:
        df: Lote a validar.
        contract: Contrato a aplicar.

    Returns:
        DataFrame com uma linha por violação (`column`, `check`,
        `failure_case`, `index`). Regras de linha inteira (ex.: duplicatas)
        aparecem uma vez por registro, com `column = "<row>"` e o
        `application_id` em `failure_case`. Vazio se o lote estiver válido.
    """
    cols = ["column", "check", "failure_case", "index"]
    try:
        contract.validate(df, lazy=True)
    except SchemaErrors as err:
        return _collapse_row_checks(err.failure_cases, df)[cols].reset_index(drop=True)
    return pd.DataFrame(columns=cols)


def _collapse_row_checks(failures: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """Reduz falhas de regras de linha inteira a uma ocorrência por registro.

    O Pandera reporta checks de DataFrame repetidos para cada coluna da linha
    inválida; aqui eles viram uma única linha identificada pelo ID.

    Args:
        failures: `failure_cases` retornado pelo Pandera.
        df: Lote validado, para recuperar o `application_id`.

    Returns:
        `failure_cases` sem as repetições por coluna.
    """
    row_level = (failures["schema_context"] == "DataFrameSchema") & failures["index"].notna()
    rows = failures[row_level].drop_duplicates(["check", "index"]).copy()
    rows["column"] = "<row>"
    if ID_COLUMN in df.columns:
        rows["failure_case"] = df.loc[rows["index"].astype(int), ID_COLUMN].to_numpy()
    return pd.concat([failures[~row_level], rows], ignore_index=True)


def validate(df: pd.DataFrame, contract: pa.DataFrameSchema) -> pd.DataFrame:
    """Valida um lote e interrompe o fluxo em caso de violação.

    Todas as regras são avaliadas (`lazy=True`) antes de falhar, para que o
    erro liste todas as violações de uma vez.

    Args:
        df: Lote a validar.
        contract: Contrato a aplicar.

    Returns:
        O próprio DataFrame validado.

    Raises:
        SchemaErrors: Se houver qualquer violação do contrato.
    """
    try:
        return contract.validate(df, lazy=True)
    except SchemaErrors as err:
        summary = err.failure_cases.groupby(["column", "check"], dropna=False).size()
        logger.error("Contrato '%s' violado:\n%s", contract.name, summary.to_string())
        raise
