"""Métricas de avaliação para modelos de credit scoring.

Além das métricas usuais de classificação, inclui as métricas de mercado para
crédito (KS e Gini) e o custo esperado definido pelo próprio German Credit:
aprovar um mau pagador (falso negativo) custa 5x mais que recusar um bom
pagador (falso positivo).

Estas funções são reutilizadas no monitoramento, para comparar o desempenho
do modelo em produção com o desempenho de referência do treino.
"""

from __future__ import annotations

from typing import Final

import numpy as np
import pandas as pd
from sklearn.metrics import (
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

COST_FN: Final[float] = 5.0
COST_FP: Final[float] = 1.0


def ks_statistic(y_true: pd.Series | np.ndarray, y_proba: np.ndarray) -> float:
    """Calcula a estatística KS (Kolmogorov-Smirnov) de separação.

    É a maior distância entre as distribuições acumuladas de score de bons e
    maus pagadores, equivalente a `max(TPR - FPR)` na curva ROC.

    Args:
        y_true: Alvo real (1 = mau pagador).
        y_proba: Probabilidade prevista de default.

    Returns:
        KS entre 0 e 1 (quanto maior, melhor a separação).
    """
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    return float(np.max(tpr - fpr))


def expected_cost(
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
    cost_fn: float = COST_FN,
    cost_fp: float = COST_FP,
) -> float:
    """Calcula o custo médio por solicitação segundo a matriz de custos.

    Args:
        y_true: Alvo real (1 = mau pagador).
        y_pred: Classe prevista (1 = recusar crédito).
        cost_fn: Custo de aprovar um mau pagador.
        cost_fp: Custo de recusar um bom pagador.

    Returns:
        Custo médio por solicitação.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return float((cost_fn * fn + cost_fp * fp) / len(y_pred))


def find_best_threshold(
    y_true: pd.Series | np.ndarray,
    y_proba: np.ndarray,
    cost_fn: float = COST_FN,
    cost_fp: float = COST_FP,
) -> float:
    """Encontra o ponto de corte que minimiza o custo esperado.

    Deve ser aplicado sobre predições out-of-fold do treino, nunca sobre o
    teste, para não vazar informação na avaliação final.

    Args:
        y_true: Alvo real (1 = mau pagador).
        y_proba: Probabilidade prevista de default.
        cost_fn: Custo de aprovar um mau pagador.
        cost_fp: Custo de recusar um bom pagador.

    Returns:
        Ponto de corte: probabilidades `>=` a ele resultam em recusa.
    """
    thresholds = np.round(np.arange(0.01, 1.0, 0.01), 2)
    costs = [
        expected_cost(y_true, (y_proba >= t).astype(int), cost_fn, cost_fp) for t in thresholds
    ]
    return float(thresholds[int(np.argmin(costs))])


def classification_report(
    y_true: pd.Series | np.ndarray, y_proba: np.ndarray, threshold: float
) -> dict[str, float]:
    """Gera o conjunto completo de métricas de avaliação.

    Args:
        y_true: Alvo real (1 = mau pagador).
        y_proba: Probabilidade prevista de default.
        threshold: Ponto de corte para converter probabilidade em decisão.

    Returns:
        Dicionário com `auc`, `gini`, `ks`, `brier`, `precision`, `recall`,
        `f1`, `expected_cost` e `approval_rate`.
    """
    y_pred = (y_proba >= threshold).astype(int)
    auc = roc_auc_score(y_true, y_proba)
    return {
        "auc": float(auc),
        "gini": float(2 * auc - 1),
        "ks": ks_statistic(y_true, y_proba),
        "brier": float(brier_score_loss(y_true, y_proba)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "expected_cost": expected_cost(y_true, y_pred),
        "approval_rate": float(1 - y_pred.mean()),
    }
