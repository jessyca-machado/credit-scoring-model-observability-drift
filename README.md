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
> usados apenas para auditoria de viés, não como features. Ver [Proteção de dados](#proteção-de-dados-e-governança-lgpd).

## Proteção de dados e governança (LGPD)

> **Escopo.** O German Credit é um dataset público e anonimizado (Alemanha, 1994), sem titulares
> identificáveis. Este plano descreve como o sistema trata os dados **como se operasse com clientes
> reais no Brasil**. Cada medida indica se está ✅ implementada ou 📋 apenas planejada.

### 1. Base legal

| Tratamento | Base legal (LGPD) |
| --- | --- |
| Análise de risco e decisão de crédito | **Art. 7º, X: proteção do crédito.** Não depende de consentimento; o tratamento se limita à finalidade de avaliar o risco de concessão |
| Monitoramento do modelo (drift, desempenho) | Art. 7º, IX: legítimo interesse, para garantir que as decisões continuem corretas. É compatível com a finalidade original (art. 6º, I) |
| Auditoria de viés com atributos protegidos | Art. 6º, IX: princípio da não discriminação. Os atributos servem **só** para medir viés, nunca para decidir |

**Decisão automatizada (art. 20).** O titular pode pedir a revisão de uma decisão automatizada e
informações sobre os critérios usados. ✅ Toda predição guarda o `application_id`, a versão do modelo,
o threshold e o horário (`data/production/predictions/`). Isso permite reconstruir e reavaliar qualquer
decisão. 📋 Falta um canal de revisão com análise humana e uma explicação individual por decisão (ex.: SHAP).

> Sexo, idade, estado civil e nacionalidade **não** são "dados sensíveis" no sentido estrito do
> art. 5º, II, mas são atributos de **potencial discriminatório** e recebem tratamento equivalente.

### 2. Tratamento de dados pessoais (PII)

| Medida | Como foi feito | |
| --- | --- | --- |
| **Minimização** (art. 6º, III) | Nenhum identificador direto (nome, CPF, endereço). O contrato de dados é estrito e **rejeita colunas não previstas**: um lote com `cpf` é bloqueado | ✅ |
| **Pseudonimização** | `application_id` (`APP-00001`) é sequencial e não deriva de dado pessoal. Ele liga predição, rótulo e lote sem expor o titular | ✅ |
| **Separação de atributos protegidos** | `sex`, `marital_status`, `foreign_worker` e `age` ficam em `SENSITIVE_COLUMNS` e são excluídos das features do modelo | ✅ |
| **Saídas sem PII** | O log de predições, a resposta da API, os logs (Loki) e as métricas (Prometheus) contêm apenas ID, resultado do modelo e agregados | ✅ |
| **Dados fora do Git** | `data/`, `models/`, `reports/` e `logs/` estão no `.gitignore` | ✅ |
| **Qualidade** (art. 6º, V) | Contrato de dados com Pandera: lotes inválidos vão para quarentena e não chegam ao modelo | ✅ |
| **Segurança** (art. 46) | Criptografia em repouso, controle de acesso aos buckets e à API, autenticação no Grafana | 📋 |
| **RIPD** (art. 38) | Relatório de Impacto à Proteção de Dados formal | 📋 |

### 3. Plano de retenção

| Dado | Prazo | Justificativa | Descarte |
| --- | --- | --- | --- |
| Solicitações e rótulos de produção (`data/production/`) | 5 anos após o fim da relação | Prazo prescricional de dívidas (CC art. 206, §5º, I) e limite para informação negativa (CDC art. 43, §1º) | Exclusão ou anonimização |
| Log de predições (auditoria de decisões) | 5 anos | Atender pedidos de revisão (art. 20) e auditorias | Exclusão |
| Lotes em quarentena | 90 dias | Tempo para investigar e corrigir a origem | Exclusão |
| Dataset de referência e modelo | Enquanto o modelo estiver em produção + 5 anos | Reproduzir decisões tomadas com aquela versão | Exclusão |
| Logs operacionais (Loki) | 6 meses | Diagnóstico de incidentes; referência no Marco Civil (art. 15) | Expiração automática |
| Métricas agregadas (Prometheus) | 1 ano | Sem dado pessoal; análise de tendência de drift | Expiração automática |

📋 Os prazos ainda não são aplicados automaticamente. A implementação prevista é um job agendado de
expurgo, mais `retention_period` no Loki e `--storage.tsdb.retention.time` no Prometheus. Pedidos de
eliminação do titular (art. 18, VI) devem respeitar esses prazos de guarda legal.

### 4. Vieses: avaliação e mitigação

Excluir os atributos protegidos **não garante** um modelo justo: outras variáveis podem funcionar
como *proxy* deles. Por isso medimos a taxa de aprovação por grupo no holdout de teste (n = 200),
usando a **razão de impacto desigual** (aprovação do grupo menos favorecido ÷ do mais favorecido).
Pela regra dos 4/5, valores abaixo de **0,8** indicam possível discriminação.

| Grupo | Aprovação | Aprovação entre bons pagadores | Default real | Razão de impacto |
| --- | --- | --- | --- | --- |
| Masculino / Feminino | 34,5% / 32,8% | 44% / 42% | 29% / 33% | **0,95** ✅ |
| ≥ 25 anos / < 25 anos | 36,6% / 23,1% | 45% / 35% | 26% / 49% | **0,63** ⚠️ |
| Estrangeiro: não / sim | 50,0% / 33,5% | — | — | 0,67 (n = 6, inconclusivo) |

**Leitura.** Não há disparidade relevante por sexo. Clientes **com menos de 25 anos** são aprovados
bem menos, embora `age` não seja feature. Parte da diferença reflete um risco real maior (default de
49% contra 26%), mas, entre os **bons pagadores**, os jovens também são menos aprovados (35% contra
45%). Isso é um sinal de **viés por proxy**, provavelmente via `employment_since`, `housing` e
`duration_months`.

**Mitigação:**

1. ✅ Atributos protegidos fora das features; ✅ medição de viés documentada.
2. 📋 Monitorar a razão de impacto por grupo a cada lote, como métrica no Prometheus com alerta abaixo de 0,8, no mesmo fluxo do drift.
3. 📋 Revisão humana obrigatória para recusas de clientes < 25 anos próximas do threshold.
4. 📋 No próximo retreino, testar a remoção ou o agrupamento dos proxies e restrições de fairness (ex.: `fairlearn`), medindo o custo em AUC.
5. 📋 Registrar a decisão final (aceitar, mitigar ou rejeitar o modelo) no Model Card, com aprovação de risco e jurídico.

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
├── docker-compose.yml      # api, prometheus, pushgateway, loki, alloy, grafana, mlflow
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
