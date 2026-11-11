# credit-scoring-model-observability-drift
Quarto tech challenge para o curso pós-tech da FIAP de machine learning engineering - construção de um modelo de classificação simples, com contratos de qualidade de dados, simulação de um cenário de degradação (Data Drift e Concept Drift), uso de ferramentas de observabilidade e doc sob as regras de governança e privacidade (LGPD).

## Dataset

[Statlog German Credit Data](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data) (UCI, licença CC BY 4.0):
1.000 solicitantes de crédito, 20 atributos (7 numéricos, 13 categóricos) e alvo binário
(700 bons / 300 maus pagadores). O dado bruto não é versionado; baixe com `make data`
(integridade validada por SHA-256). A carga decodificada fica em
`credit_scoring.data.load.load_german_credit()`, com o alvo `default` (1 = mau pagador).

> O dataset original define custo assimétrico: classificar um mau pagador como bom custa 5x mais
> que o erro inverso. Atributos sensíveis (`sex`, `marital_status`, `foreign_worker`, `age`) são
> usados apenas para auditoria de viés, não como features.

## Estrutura do projeto

```
├── configs/                # parâmetros (thresholds de drift, caminhos, seeds)
├── data/
│   ├── raw/                # dados brutos (não versionados - LGPD)
│   ├── processed/          # dados tratados
│   ├── reference/          # baseline do treino para comparação de drift
│   └── production/         # lotes simulados de produção ao longo do tempo
├── models/                 # artefatos treinados (não versionados)
├── notebooks/              # EDA e análise causal exploratória
├── scripts/                # utilitários e automações
├── src/credit_scoring/
│   ├── data/               # ingestão e pré-processamento
│   ├── contracts/          # contratos de qualidade de dados (Pandera) - ver docs/governance/data_contract.md
│   ├── model/              # treino, avaliação e registro (MLflow)
│   ├── drift/              # simulação e detecção de Data/Concept Drift
│   ├── causal/             # análise causal (DAG + DoWhy)
│   ├── observability/      # logs JSON (Loki) e métricas das etapas (Pushgateway)
│   └── serving/            # API FastAPI + endpoint /metrics (Prometheus)
├── monitoring/
│   ├── prometheus/         # scrape config e regras de alerta
│   ├── alloy/              # coleta de logs/*.jsonl -> Loki
│   └── grafana/            # datasources (Prometheus + Loki) e dashboard provisionado
├── docs/
│   ├── governance/         # LGPD, RIPD, base legal, retenção
│   ├── model_card/         # Model Card e Data Card
│   └── runbooks/           # observabilidade e resposta a alertas (observability.md)
├── tests/{unit,integration}/
├── docker-compose.yml      # api + prometheus + grafana + mlflow
├── Dockerfile
├── Makefile
└── pyproject.toml
```

## Como rodar

```bash
make install    # uv sync - cria .venv e instala dependências do uv.lock
make data       # baixa o German Credit (UCI) para data/raw
make preprocess # trata, valida contra o contrato de dados e gera train/test e reference
make demo-contract  # demonstra o contrato bloqueando um lote com erros
make ingest FILE=… # ingere um lote: aceito ou quarentena (exit 1)
make train      # treina o XGBoost (busca de hiperparâmetros + threshold por custo) e salva em models/
make mlflow-ui  # abre o MLflow em http://localhost:5000 para comparar execuções
make simulate   # gera 12 lotes mensais de produção (jan-dez/2025) com data drift (abr+) e concept drift (ago+)
make predict    # gera PD, score (0-1000) e decisão para cada lote de produção
make detect     # Evidently: referência x produção por mês (drift PSI + desempenho) -> reports/
make pipeline   # roda todas as etapas instrumentadas (logs + métricas) com o mesmo RUN_ID
make serve      # API local: POST /predict, GET /health, GET /metrics
make up         # sobe api (8000), prometheus (9090), pushgateway (9091), loki, alloy, grafana (3000) e mlflow (5000)
make traffic    # gera tráfego na API a partir dos lotes de produção
make observability  # up + pipeline + traffic; dashboard em http://localhost:3000
make test
```
