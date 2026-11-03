# credit-scoring-model-observability-drift
Quarto tech challenge para o curso pós-tech da FIAP de machine learning engineering - construção de um modelo de classificação simples, com contratos de qualidade de dados, simulação de um cenário de degradação (Data Drift e Concept Drift), uso de ferramentas de observabilidade e doc sob as regras de governança e privacidade (LGPD).

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
│   ├── contracts/          # contratos de qualidade de dados (Pandera)
│   ├── model/              # treino, avaliação e registro (MLflow)
│   ├── drift/              # simulação e detecção de Data/Concept Drift
│   ├── causal/             # análise causal (DAG + DoWhy)
│   └── serving/            # API FastAPI + endpoint /metrics (Prometheus)
├── monitoring/
│   ├── prometheus/         # scrape config e regras de alerta
│   └── grafana/            # datasources e dashboards provisionados
├── docs/
│   ├── governance/         # LGPD, RIPD, base legal, retenção
│   ├── model_card/         # Model Card e Data Card
│   └── runbooks/           # resposta a incidentes de drift
├── tests/{unit,integration}/
├── docker-compose.yml      # api + prometheus + grafana + mlflow
├── Dockerfile
├── Makefile
└── pyproject.toml
```

## Como rodar

```bash
make install    # uv sync - cria .venv e instala dependências do uv.lock
make train      # treina o modelo
make simulate   # gera lotes com Data/Concept Drift
make detect     # calcula métricas de drift
make up         # sobe api, prometheus (9090), grafana (3000) e mlflow (5000)
make test
```
