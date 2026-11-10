"""Carga e decodificação do German Credit Data.

O arquivo original usa códigos simbólicos (ex.: A11, A34). Aqui eles são
traduzidos para rótulos legíveis e o alvo é convertido para o padrão de
credit scoring: `default = 1` (mau pagador) e `default = 0` (bom pagador).

Observação LGPD: `personal_status` combina sexo e estado civil (dado pessoal
com potencial discriminatório). Ele é separado em `sex` e `marital_status`
para permitir análises de fairness, e listado em `SENSITIVE_COLUMNS` para que
possa ser excluído das features do modelo.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import pandas as pd

from credit_scoring.data.download import DEFAULT_PATH

TARGET: Final[str] = "default"
ID_COLUMN: Final[str] = "application_id"

RAW_COLUMNS: Final[list[str]] = [
    "checking_status",
    "duration_months",
    "credit_history",
    "purpose",
    "credit_amount",
    "savings_status",
    "employment_since",
    "installment_rate",
    "personal_status",
    "other_debtors",
    "residence_since",
    "property",
    "age",
    "other_installment_plans",
    "housing",
    "existing_credits",
    "job",
    "num_dependents",
    "own_telephone",
    "foreign_worker",
    "class",
]

CATEGORY_MAPS: Final[dict[str, dict[str, str]]] = {
    "checking_status": {
        "A11": "< 0 DM",
        "A12": "0 <= ... < 200 DM",
        "A13": ">= 200 DM",
        "A14": "no checking account",
    },
    "credit_history": {
        "A30": "no credits / all paid duly",
        "A31": "all credits at this bank paid duly",
        "A32": "existing credits paid duly",
        "A33": "delay in past",
        "A34": "critical account / other credits",
    },
    "purpose": {
        "A40": "car (new)",
        "A41": "car (used)",
        "A42": "furniture/equipment",
        "A43": "radio/television",
        "A44": "domestic appliances",
        "A45": "repairs",
        "A46": "education",
        "A47": "vacation",
        "A48": "retraining",
        "A49": "business",
        "A410": "others",
    },
    "savings_status": {
        "A61": "< 100 DM",
        "A62": "100 <= ... < 500 DM",
        "A63": "500 <= ... < 1000 DM",
        "A64": ">= 1000 DM",
        "A65": "unknown / no savings",
    },
    "employment_since": {
        "A71": "unemployed",
        "A72": "< 1 year",
        "A73": "1 <= ... < 4 years",
        "A74": "4 <= ... < 7 years",
        "A75": ">= 7 years",
    },
    "other_debtors": {
        "A101": "none",
        "A102": "co-applicant",
        "A103": "guarantor",
    },
    "property": {
        "A121": "real estate",
        "A122": "savings agreement / life insurance",
        "A123": "car or other",
        "A124": "unknown / no property",
    },
    "other_installment_plans": {
        "A141": "bank",
        "A142": "stores",
        "A143": "none",
    },
    "housing": {
        "A151": "rent",
        "A152": "own",
        "A153": "for free",
    },
    "job": {
        "A171": "unemployed / unskilled non-resident",
        "A172": "unskilled resident",
        "A173": "skilled employee / official",
        "A174": "management / self-employed / highly qualified",
    },
    "own_telephone": {"A191": "no", "A192": "yes"},
    "foreign_worker": {"A201": "yes", "A202": "no"},
}

# personal_status -> (sex, marital_status)
PERSONAL_STATUS_MAP: Final[dict[str, tuple[str, str]]] = {
    "A91": ("male", "divorced/separated"),
    "A92": ("female", "divorced/separated/married"),
    "A93": ("male", "single"),
    "A94": ("male", "married/widowed"),
    "A95": ("female", "single"),
}

# Atributos sensíveis/protegidos: não devem ser usados como features do modelo,
# apenas para auditoria de viés (LGPD art. 6º, IX - não discriminação).
SENSITIVE_COLUMNS: Final[list[str]] = ["sex", "marital_status", "foreign_worker", "age"]


def category_dtype(column: str) -> pd.CategoricalDtype:
    """Retorna o dtype categórico com o domínio completo de uma coluna.

    O domínio vem do dicionário de dados da UCI, e não dos valores observados.
    Assim, categorias válidas mas ausentes na amostra (ex.: `purpose = vacation`)
    continuam conhecidas, e lotes de produção mantêm o mesmo dtype do treino.

    Args:
        column: Nome de uma coluna categórica decodificada (inclui `sex` e
            `marital_status`).

    Returns:
        `CategoricalDtype` não ordenado com todas as categorias válidas.

    Raises:
        KeyError: Se `column` não for uma coluna categórica conhecida.
    """
    if column in CATEGORY_MAPS:
        categories = list(CATEGORY_MAPS[column].values())
    elif column in ("sex", "marital_status"):
        idx = 0 if column == "sex" else 1
        categories = list(dict.fromkeys(v[idx] for v in PERSONAL_STATUS_MAP.values()))
    else:
        raise KeyError(f"Coluna categórica desconhecida: {column}")
    return pd.CategoricalDtype(categories=categories, ordered=False)


def load_raw(path: Path | str = DEFAULT_PATH) -> pd.DataFrame:
    """Lê o arquivo original do German Credit sem nenhuma transformação.

    O arquivo da UCI é separado por espaço e não possui cabeçalho; os nomes
    das colunas são atribuídos a partir de `RAW_COLUMNS`.

    Args:
        path: Caminho do arquivo `german.data`. Padrão: `data/raw/german.data`.

    Returns:
        DataFrame com 21 colunas: 20 atributos com códigos simbólicos
        (ex.: `A11`) e a coluna `class` (1 = bom, 2 = mau pagador).

    Raises:
        FileNotFoundError: Se o arquivo não existir. Rode `make data` antes.
    """
    return pd.read_csv(path, sep=" ", header=None, names=RAW_COLUMNS)


def decode(df: pd.DataFrame) -> pd.DataFrame:
    """Decodifica o German Credit bruto para um formato legível e modelável.

    Transformações aplicadas:
        - Códigos simbólicos viram rótulos legíveis (`CATEGORY_MAPS`), com
          dtype `category`.
        - `personal_status` é separado em `sex` e `marital_status`.
        - `class` é convertido no alvo `default` (1 = mau pagador, 0 = bom).

    Args:
        df: DataFrame no formato retornado por `load_raw`. Não é modificado.

    Returns:
        Novo DataFrame com 22 colunas: 21 atributos e o alvo `default` (int8).
    """
    out = df.copy()
    for col, mapping in CATEGORY_MAPS.items():
        out[col] = out[col].map(mapping).astype(category_dtype(col))

    status = out.pop("personal_status").map(PERSONAL_STATUS_MAP)
    out["sex"] = status.str[0].astype(category_dtype("sex"))
    out["marital_status"] = status.str[1].astype(category_dtype("marital_status"))

    # Original: 1 = bom, 2 = mau -> default: 0 = bom, 1 = mau
    out[TARGET] = (out.pop("class") == 2).astype("int8")
    return out


def load_german_credit(path: Path | str = DEFAULT_PATH) -> pd.DataFrame:
    """Carrega o German Credit decodificado e pronto para uso.

    Atalho para `decode(load_raw(path))`.

    Args:
        path: Caminho do arquivo `german.data`. Padrão: `data/raw/german.data`.

    Returns:
        DataFrame com 1.000 linhas e 22 colunas, incluindo o alvo `default`.

    Raises:
        FileNotFoundError: Se o arquivo não existir. Rode `make data` antes.

    Example:
        >>> df = load_german_credit()
        >>> df.shape
        (1000, 22)
    """
    return decode(load_raw(path))
