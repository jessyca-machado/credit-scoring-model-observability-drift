"""Simulação de um "Dataset de Produção" ao longo do tempo com drift.

Cenário econômico: a partir de um período estável, a fintech passa a operar
num ambiente de inflação alta e captação de novos perfis de clientes.

Fases (controladas por `configs/config.yaml` -> `simulation`):

1. **Estável** (lotes `< data_drift_start`): mesma distribuição do treino.
2. **Data drift** (a partir de `data_drift_start`, intensidade crescente) -
   muda P(X), a distribuição das variáveis:
     - `credit_amount`: inflação mensal composta (valores nominais maiores);
     - `duration_months`: parte dos clientes alonga o prazo para caber a parcela;
     - `checking_status`, `credit_history`, `employment_since`: mais clientes
       com conta negativa, sem histórico de crédito e empregos recentes
       (novo perfil captado pela fintech).
3. **Concept drift** (a partir de `concept_drift_start`, intensidade crescente) -
   muda P(y | X), a relação entre variáveis e inadimplência:
     - clientes **sem conta corrente**, antes o perfil mais seguro (10% de
       default), passam a inadimplir mais (desbancarizados atingidos pela crise);
     - clientes **com créditos em outras instituições** (`credit_history =
       critical account / other credits`), também tidos como seguros no treino
       (17% de default), passam a inadimplir por superendividamento.

   Ambos os perfis são considerados de **baixo risco** pelo modelo: o concept
   drift contradiz o que ele aprendeu, e por isso degrada a discriminação.

Os lotes são gerados a partir do conjunto de **teste** (nunca visto pelo
modelo), com reamostragem ponderada e pequeno ruído em `credit_amount` para
que não haja registros duplicados.

Saídas:
    - `data/production/incoming/prod_AAAA_MM.parquet`: lote sem rótulo, que
      passa pela ingestão (contrato de dados) e termina em `accepted/`;
    - `data/production/labels/prod_AAAA_MM.parquet`: rótulos tardios
      (`application_id`, `default`);
    - `data/production/simulation_manifest.json`: "gabarito" do que foi
      alterado em cada lote, para validar o diagnóstico e a análise causal.

Uso:
    uv run python -m credit_scoring.drift.simulate
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from credit_scoring.config import load_config
from credit_scoring.data.ingest import ingest_batch
from credit_scoring.data.load import ID_COLUMN, TARGET

logger = logging.getLogger(__name__)

STRESSED_CHECKING: tuple[str, ...] = ("< 0 DM",)
FIRST_TIME_HISTORY: tuple[str, ...] = (
    "no credits / all paid duly",
    "all credits at this bank paid duly",
)
RECENT_EMPLOYMENT: tuple[str, ...] = ("< 1 year", "unemployed")


@dataclass(frozen=True)
class BatchManifest:
    """Registro do que foi aplicado em um lote (gabarito da simulação).

    Attributes:
        batch: Nome do lote (`prod_AAAA_MM`).
        month: Mês de referência (`AAAA-MM`).
        data_drift: Intensidade do data drift (0 a 1).
        concept_drift: Intensidade do concept drift (0 a 1).
        inflation_factor: Multiplicador acumulado aplicado a `credit_amount`.
        n_rows: Quantidade de solicitações.
        default_rate: Taxa de inadimplência real do lote.
        labels_flipped: Bons pagadores convertidos em maus pelo concept drift.
    """

    batch: str
    month: str
    data_drift: float
    concept_drift: float
    inflation_factor: float
    n_rows: int
    default_rate: float
    labels_flipped: int


def ramp(t: int, start: int, n_batches: int) -> float:
    """Intensidade linear de um efeito: 0 antes de `start`, 1 no último lote.

    Args:
        t: Índice do lote (0-based).
        start: Lote em que o efeito começa (0-based).
        n_batches: Total de lotes.

    Returns:
        Intensidade entre 0 e 1.
    """
    if t < start:
        return 0.0
    return float(min(1.0, (t - start + 1) / max(1, n_batches - start)))


def sample_weights(pool: pd.DataFrame, intensity: float, profile_weight: float) -> np.ndarray:
    """Calcula pesos de amostragem que favorecem o novo perfil de cliente.

    Cada característica do novo perfil (conta negativa, sem histórico,
    emprego recente) soma `profile_weight * intensity` ao peso da linha.

    Args:
        pool: Base de onde os clientes são amostrados.
        intensity: Intensidade do data drift (0 a 1).
        profile_weight: Peso extra de cada característica com intensidade 1.

    Returns:
        Vetor de probabilidades (soma 1).
    """
    w = 1.0 + profile_weight * intensity * (
        pool["checking_status"].isin(STRESSED_CHECKING).astype(float)
        + pool["credit_history"].isin(FIRST_TIME_HISTORY).astype(float)
        + pool["employment_since"].isin(RECENT_EMPLOYMENT).astype(float)
    )
    return (w / w.sum()).to_numpy()


def apply_data_drift(
    batch: pd.DataFrame,
    inflation_factor: float,
    intensity: float,
    extend_prob: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Aplica inflação, alongamento de prazo e ruído às variáveis numéricas.

    Args:
        batch: Lote amostrado.
        inflation_factor: Multiplicador acumulado de preços.
        intensity: Intensidade do data drift (0 a 1).
        extend_prob: Probabilidade máxima de alongar o prazo (com intensidade 1).
        rng: Gerador aleatório.

    Returns:
        Novo lote com `credit_amount`, `duration_months` e
        `monthly_installment` atualizados.
    """
    out = batch.copy()
    noise = rng.uniform(0.95, 1.05, len(out))
    out["credit_amount"] = np.maximum(
        1, np.round(out["credit_amount"] * inflation_factor * noise)
    ).astype("int64")

    extend = rng.random(len(out)) < extend_prob * intensity
    extra = rng.choice([6, 12], size=len(out))
    out["duration_months"] = np.where(
        extend, np.minimum(out["duration_months"] + extra, 72), out["duration_months"]
    ).astype("int64")

    out["monthly_installment"] = (out["credit_amount"] / out["duration_months"]).round(2)
    return out


def apply_concept_drift(
    batch: pd.DataFrame,
    intensity: float,
    flip_no_account: float,
    flip_overindebted: float,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, int]:
    """Converte bons pagadores em maus em perfis afetados pela crise.

    Altera P(default | X) sem alterar X: o mesmo perfil passa a ter risco
    diferente do aprendido pelo modelo.

    Args:
        batch: Lote com o alvo `default`.
        intensity: Intensidade do concept drift (0 a 1).
        flip_no_account: Probabilidade máxima de default para clientes bons
            sem conta corrente.
        flip_overindebted: Probabilidade máxima de default para clientes bons
            com créditos em outras instituições.
        rng: Gerador aleatório.

    Returns:
        Tupla `(lote, quantidade de rótulos alterados)`.
    """
    out = batch.copy()
    good = out[TARGET] == 0
    no_account = out["checking_status"] == "no checking account"
    overindebted = out["credit_history"] == "critical account / other credits"

    prob = np.where(no_account, flip_no_account, 0.0)
    prob = np.maximum(prob, np.where(overindebted, flip_overindebted, 0.0)) * intensity
    flip = good.to_numpy() & (rng.random(len(out)) < prob)

    out.loc[flip, TARGET] = 1
    out[TARGET] = out[TARGET].astype("int8")
    return out, int(flip.sum())


def generate_batch(
    pool: pd.DataFrame, t: int, sim: dict[str, Any], rng: np.random.Generator
) -> tuple[pd.DataFrame, BatchManifest]:
    """Gera um lote mensal de produção com rótulos.

    Args:
        pool: Base de clientes (conjunto de teste).
        t: Índice do lote (0-based).
        sim: Seção `simulation` do config.
        rng: Gerador aleatório.

    Returns:
        Tupla `(lote com rótulo, manifesto do lote)`.
    """
    n = sim["n_batches"]
    month = (pd.Period(sim["start_month"], "M") + t).strftime("%Y-%m")
    name = f"prod_{month.replace('-', '_')}"
    data_int = ramp(t, sim["data_drift_start"], n)
    concept_int = ramp(t, sim["concept_drift_start"], n)
    months_inflating = max(0, t - sim["data_drift_start"] + 1)
    inflation = (1 + sim["monthly_inflation"]) ** months_inflating

    idx = rng.choice(
        len(pool),
        size=sim["batch_size"],
        replace=True,
        p=sample_weights(pool, data_int, sim["profile_weight"]),
    )
    batch = pool.iloc[idx].reset_index(drop=True)
    batch = apply_data_drift(batch, inflation, data_int, sim["extend_duration_prob"], rng)
    batch, flipped = apply_concept_drift(
        batch, concept_int, sim["flip_no_account"], sim["flip_overindebted"], rng
    )

    batch = batch.drop_duplicates(subset=batch.columns.drop([ID_COLUMN, TARGET]))
    batch[ID_COLUMN] = [f"APP-{(t + 1) * 100_000 + i}" for i in range(len(batch))]
    batch = batch.reset_index(drop=True)

    manifest = BatchManifest(
        batch=name,
        month=month,
        data_drift=round(data_int, 3),
        concept_drift=round(concept_int, 3),
        inflation_factor=round(inflation, 4),
        n_rows=len(batch),
        default_rate=round(float(batch[TARGET].mean()), 4),
        labels_flipped=flipped,
    )
    return batch, manifest


def simulate(cfg: dict[str, Any]) -> list[BatchManifest]:
    """Gera todos os lotes, ingere-os pelo contrato e salva rótulos e manifesto.

    Args:
        cfg: Configurações do projeto.

    Returns:
        Manifestos de todos os lotes.

    Raises:
        RuntimeError: Se algum lote for bloqueado pelo contrato de dados.
    """
    sim = cfg["simulation"]
    prod = Path(cfg["paths"]["production"])
    pool = pd.read_parquet(Path(cfg["paths"]["processed"]) / "test.parquet")
    rng = np.random.default_rng(cfg["seed"])

    incoming, labels_dir = prod / "incoming", prod / "labels"
    for d in (incoming, labels_dir, prod / "accepted"):
        d.mkdir(parents=True, exist_ok=True)
        for old in d.glob("prod_*.parquet"):
            old.unlink()

    manifests: list[BatchManifest] = []
    for t in range(sim["n_batches"]):
        batch, manifest = generate_batch(pool, t, sim, rng)
        path = incoming / f"{manifest.batch}.parquet"
        batch.drop(columns=TARGET).to_parquet(path, index=False)
        batch[[ID_COLUMN, TARGET]].to_parquet(labels_dir / path.name, index=False)

        if not ingest_batch(path, prod).accepted:
            raise RuntimeError(f"Lote simulado {path.name} violou o contrato de dados")
        manifests.append(manifest)

    (prod / "simulation_manifest.json").write_text(
        json.dumps(
            {"config": sim, "batches": [asdict(m) for m in manifests]},
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return manifests


def main() -> None:
    """Executa a simulação conforme `configs/config.yaml` e exibe o resumo."""
    manifests = simulate(load_config())
    logger.info("%-15s %-8s %6s %8s %9s %8s %7s", "lote", "mês", "data", "concept",
                "inflação", "default", "flips")
    for m in manifests:
        logger.info(
            "%-15s %-8s %6.2f %8.2f %9.3f %7.1f%% %7d",
            m.batch,
            m.month,
            m.data_drift,
            m.concept_drift,
            m.inflation_factor,
            100 * m.default_rate,
            m.labels_flipped,
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
