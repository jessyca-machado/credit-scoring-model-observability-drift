"""Treino do classificador XGBoost de credit scoring.

Fluxo:
    1. Lê `train.parquet` e `test.parquet` gerados por `make preprocess`.
    2. Busca hiperparâmetros com validação cruzada estratificada (AUC).
    3. Escolhe o ponto de corte que minimiza o custo esperado (FN = 5, FP = 1)
       usando predições out-of-fold do treino.
    4. Avalia uma única vez no teste.
    5. Salva o pipeline e os metadados em `models/` e registra no MLflow.

Uso:
    uv run python -m credit_scoring.model.train
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import mlflow
import numpy as np
import pandas as pd
import sklearn
import xgboost
from mlflow.models import infer_signature
from scipy.stats import loguniform, randint, uniform
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

from credit_scoring.config import load_config
from credit_scoring.data.load import TARGET
from credit_scoring.data.preprocess import CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES
from credit_scoring.model.evaluate import classification_report, find_best_threshold

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
logger = logging.getLogger(__name__)

SKOPS_TRUSTED_TYPES: list[str] = ["xgboost.core.Booster", "xgboost.sklearn.XGBClassifier"]

PARAM_DISTRIBUTIONS: dict[str, Any] = {
    "clf__n_estimators": randint(100, 600),
    "clf__max_depth": randint(2, 6),
    "clf__learning_rate": loguniform(0.01, 0.2),
    "clf__subsample": uniform(0.6, 0.4),
    "clf__colsample_bytree": uniform(0.5, 0.5),
    "clf__min_child_weight": randint(1, 10),
    "clf__reg_lambda": loguniform(0.1, 10),
}


def build_pipeline(min_frequency: int = 10, seed: int = 42) -> Pipeline:
    """Monta o pipeline de pré-processamento + XGBoost.

    Categóricas passam por one-hot encoding; categorias com menos de
    `min_frequency` ocorrências no treino são agrupadas em "infrequent", e
    categorias inéditas em produção caem nesse mesmo grupo em vez de gerar
    erro. Numéricas seguem sem transformação (árvores não exigem escala).

    Args:
        min_frequency: Frequência mínima para uma categoria ter coluna própria.
        seed: Semente do XGBoost.

    Returns:
        `Pipeline` do scikit-learn ainda não treinado.
    """
    preprocessor = ColumnTransformer(
        [
            (
                "cat",
                OneHotEncoder(
                    handle_unknown="infrequent_if_exist",
                    min_frequency=min_frequency,
                    sparse_output=False,
                ),
                CATEGORICAL_FEATURES,
            ),
            ("num", "passthrough", NUMERIC_FEATURES),
        ],
        verbose_feature_names_out=False,
    )
    clf = XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        tree_method="hist",
        n_jobs=1,
        random_state=seed,
    )
    return Pipeline([("prep", preprocessor), ("clf", clf)])


def tune(
    pipeline: Pipeline, X: pd.DataFrame, y: pd.Series, n_iter: int, cv: StratifiedKFold, seed: int
) -> RandomizedSearchCV:
    """Busca aleatória de hiperparâmetros maximizando AUC.

    Args:
        pipeline: Pipeline base (saída de `build_pipeline`).
        X: Features de treino.
        y: Alvo de treino.
        n_iter: Número de combinações avaliadas.
        cv: Estratégia de validação cruzada.
        seed: Semente da busca.

    Returns:
        `RandomizedSearchCV` já ajustado, com `best_estimator_` re-treinado
        em todo o treino.
    """
    search = RandomizedSearchCV(
        pipeline,
        PARAM_DISTRIBUTIONS,
        n_iter=n_iter,
        scoring="roc_auc",
        cv=cv,
        random_state=seed,
        n_jobs=-1,
    )
    return search.fit(X, y)


def feature_importance(pipeline: Pipeline, top: int = 15) -> dict[str, float]:
    """Extrai a importância (gain) das features do XGBoost treinado.

    Args:
        pipeline: Pipeline treinado.
        top: Quantidade de features retornadas.

    Returns:
        Dicionário `{feature: importância}` ordenado de forma decrescente.
    """
    names = pipeline.named_steps["prep"].get_feature_names_out()
    gain = pipeline.named_steps["clf"].feature_importances_
    s = pd.Series(gain, index=names).sort_values(ascending=False).head(top)
    return {k: round(float(v), 4) for k, v in s.items()}


def file_sha256(path: Path | str) -> str:
    """Calcula o SHA-256 de um arquivo, para rastrear a linhagem dos dados.

    Args:
        path: Caminho do arquivo.

    Returns:
        Hash em hexadecimal.
    """
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def train(cfg: dict[str, Any]) -> dict[str, Any]:
    """Executa o treino completo, salva os artefatos e registra no MLflow.

    Args:
        cfg: Configurações carregadas de `configs/config.yaml`.

    Returns:
        Metadados do modelo (parâmetros, threshold, métricas, linhagem).
    """
    seed: int = cfg["seed"]
    mcfg: dict[str, Any] = cfg["model"]
    processed = Path(cfg["paths"]["processed"])
    models_dir = Path(cfg["paths"]["models"])

    train_df = pd.read_parquet(processed / "train.parquet")
    test_df = pd.read_parquet(processed / "test.parquet")
    X_train, y_train = train_df[FEATURES], train_df[TARGET]
    X_test, y_test = test_df[FEATURES], test_df[TARGET]

    cv = StratifiedKFold(n_splits=mcfg["cv_folds"], shuffle=True, random_state=seed)
    logger.info("Buscando hiperparâmetros (%d combinações x %d folds)", mcfg["n_iter"], cv.n_splits)
    search = tune(
        build_pipeline(mcfg["min_frequency"], seed), X_train, y_train, mcfg["n_iter"], cv, seed
    )
    model: Pipeline = search.best_estimator_

    oof_proba = cross_val_predict(
        model, X_train, y_train, cv=cv, method="predict_proba", n_jobs=-1
    )[:, 1]
    threshold = find_best_threshold(y_train, oof_proba, mcfg["cost_fn"], mcfg["cost_fp"])
    cv_metrics = classification_report(y_train, oof_proba, threshold)

    test_proba = model.predict_proba(X_test)[:, 1]
    test_metrics = classification_report(y_test, test_proba, threshold)

    best_params = {
        k.removeprefix("clf__"): (v.item() if isinstance(v, np.generic) else v)
        for k, v in search.best_params_.items()
    }

    metadata: dict[str, Any] = {
        "model_name": "credit-scoring-xgboost",
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "features": FEATURES,
        "target": TARGET,
        "threshold": threshold,
        "cost_matrix": {"false_negative": mcfg["cost_fn"], "false_positive": mcfg["cost_fp"]},
        "best_params": best_params,
        "metrics": {"cv_oof": cv_metrics, "test": test_metrics},
        "feature_importance": feature_importance(model),
        "data": {
            "train_rows": len(train_df),
            "test_rows": len(test_df),
            "train_default_rate": float(y_train.mean()),
            "train_sha256": file_sha256(processed / "train.parquet"),
        },
        "versions": {
            "xgboost": xgboost.__version__,
            "scikit-learn": sklearn.__version__,
            "pandas": pd.__version__,
        },
    }

    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, models_dir / "model.joblib")
    (models_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    _log_mlflow(cfg, model, metadata, X_train)
    return metadata


def _log_mlflow(
    cfg: dict[str, Any], model: Pipeline, metadata: dict[str, Any], X_train: pd.DataFrame
) -> None:
    """Registra parâmetros, métricas e o modelo no MLflow.

    Args:
        cfg: Configurações do projeto.
        model: Pipeline treinado.
        metadata: Metadados gerados no treino.
        X_train: Features de treino, para inferir a assinatura do modelo.
    """
    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["mlflow"]["experiment"])
    with mlflow.start_run(run_name=f"xgboost-{metadata['trained_at']}"):
        mlflow.log_params({**metadata["best_params"], "threshold": metadata["threshold"]})
        for split, metrics in metadata["metrics"].items():
            mlflow.log_metrics({f"{split}_{k}": v for k, v in metrics.items()})
        mlflow.set_tags({"train_sha256": metadata["data"]["train_sha256"]})
        mlflow.log_dict(metadata, "metadata.json")
        sample = X_train.head(5)
        mlflow.sklearn.log_model(
            model,
            name="model",
            signature=infer_signature(sample, model.predict_proba(sample)[:, 1]),
            input_example=sample,
            # MLflow 3 serializa com skops, que exige declarar tipos externos ao sklearn
            skops_trusted_types=SKOPS_TRUSTED_TYPES,
        )


def main() -> None:
    """Treina o modelo conforme `configs/config.yaml` e exibe as métricas."""
    metadata = train(load_config())
    m = metadata["metrics"]
    logger.info("Threshold (custo mínimo): %.2f", metadata["threshold"])
    for split in ("cv_oof", "test"):
        logger.info(
            "%-6s | AUC %.3f | Gini %.3f | KS %.3f | Recall %.3f | Custo %.3f | Aprovação %.1f%%",
            split,
            m[split]["auc"],
            m[split]["gini"],
            m[split]["ks"],
            m[split]["recall"],
            m[split]["expected_cost"],
            100 * m[split]["approval_rate"],
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
