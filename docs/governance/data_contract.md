# Contrato de Dados — Credit Scoring

Implementação: [`src/credit_scoring/contracts/schemas.py`](../../src/credit_scoring/contracts/schemas.py) (Pandera)

## Princípio

O contrato define o que é um registro **válido**, não o que é **típico**.

| Situação | Exemplo | Tratamento |
|---|---|---|
| Erro de dado | idade 15, valor negativo, categoria inexistente | **Contrato rejeita** — o lote não segue no pipeline |
| Mudança de distribuição | valores de crédito 30% maiores por inflação | **Contrato aceita** — o monitoramento de drift detecta |

Se o contrato usasse os limites observados no treino (ex.: `credit_amount ≤ 18.424`), ele confundiria drift com erro e rejeitaria exatamente os dados que precisamos monitorar.

## Contratos

| Contrato | Uso | Alvo `default` |
|---|---|---|
| `TRAINING_CONTRACT` | treino e teste (`make preprocess`) | obrigatório |
| `INFERENCE_CONTRACT` | lotes de produção | opcional — o rótulo chega meses depois da concessão |

Ambos são **estritos**: colunas não previstas são rejeitadas. Isso impede, por exemplo, que um CPF ou nome entre no pipeline por engano (LGPD, art. 6º, III — minimização).

## Regras por coluna

Nenhuma coluna aceita nulos.

| Coluna | Tipo | Regra | Justificativa |
|---|---|---|---|
| `application_id` | str | único, formato `APP-NNNNN` | rastrear predição → rótulo sem dado pessoal |
| `age` | int64 | 18 ≤ x ≤ 100 | capacidade civil (Código Civil, art. 5º) |
| `credit_amount` | int64 | 0 < x ≤ 1.000.000 | valor positivo; teto apenas de sanidade |
| `duration_months` | int64 | 1 ≤ x ≤ 120 | prazo máximo de produto |
| `monthly_installment` | float64 | > 0 | feature derivada |
| `installment_rate`, `residence_since`, `existing_credits` | int64 | ∈ {1, 2, 3, 4} | faixas codificadas pela UCI |
| `num_dependents` | int64 | ∈ {1, 2} | faixas codificadas pela UCI |
| 12 categóricas (`checking_status`, `purpose`, …) | category | ∈ domínio do dicionário UCI | categoria inexistente = erro de ingestão |
| `sex`, `marital_status` | category | ∈ domínio | dado sensível — só auditoria de viés |
| `default` | int8 | ∈ {0, 1} | 1 = mau pagador |

## Regras de linha

| Regra | Descrição |
|---|---|
| `no_duplicate_records` | duas solicitações idênticas (ignorando ID e alvo) não são aceitas |
| `monthly_installment_consistent` | `monthly_installment` = `credit_amount / duration_months` (tolerância 0,01) |
| `valid_sex_marital_status_pair` | combinação deve existir no dicionário UCI (ex.: `female` + `married/widowed` não existe) |

## Uso

```python
from credit_scoring.contracts.schemas import INFERENCE_CONTRACT, contract_report, validate

validate(df, INFERENCE_CONTRACT)               # interrompe o fluxo e lista todas as violações
report = contract_report(df, INFERENCE_CONTRACT)  # não interrompe; retorna as violações
```

Exemplo de `contract_report`:

```
column                    check                           failure_case  index
credit_scoring_inference  column_in_schema                cpf           None
credit_amount             greater_than(0)                 -100          1
age                       in_range(18, 100)               16            0
<row>                     no_duplicate_records            APP-90000     5
<row>                     monthly_installment_consistent  APP-00289     2
```

## Ingestão bloqueada pelo contrato

Todo lote de produção passa por [`ingest.py`](../../src/credit_scoring/data/ingest.py):

```
data/production/incoming/lote.parquet
        │  INFERENCE_CONTRACT
        ├── válido   → data/production/accepted/lote.parquet
        └── inválido → data/production/quarantine/lote.parquet
                       data/production/quarantine/lote.violations.csv   (exit code 1)
```

O bloqueio é do **lote inteiro**: nenhuma linha de um lote inválido chega ao modelo, pois erros em parte das linhas costumam indicar falha sistêmica na origem.

### Demonstração

```bash
make demo-contract
```

Gera um lote limpo e uma cópia com 9 tipos de erro injetados:

| Erro injetado | Regra que bloqueia |
|---|---|
| cliente com 16 anos | `age: in_range(18, 100)` |
| `credit_amount` nulo | `credit_amount: not_nullable` |
| `credit_amount` negativo | `credit_amount: greater_than(0)` |
| `purpose = "crypto"` | `purpose: isin` |
| ID repetido | `application_id: field_uniqueness` |
| solicitação duplicada com outro ID | `no_duplicate_records` |
| parcela ×10 | `monthly_installment_consistent` |
| `female` + `married/widowed` | `valid_sex_marital_status_pair` |
| coluna `cpf` | `column_in_schema` |

Resultado:

```
ACEITO    | batch_ok.parquet        |  50 linhas |  0 violações
BLOQUEADO | batch_corrupted.parquet |  51 linhas | 14 violações
```

São 14 violações para 9 erros porque alguns erros disparam mais de uma regra (ex.: `credit_amount` nulo também viola o tipo `int64` e a consistência da parcela).

Para ingerir um lote específico: `make ingest FILE=data/production/incoming/lote.parquet`.

## Observação sobre "renda não pode ser nula"

O German Credit **não possui coluna de renda**. As informações financeiras do solicitante são `credit_amount`, `installment_rate` (parcela em % da renda disponível), `checking_status` e `savings_status`, todas obrigatórias e com domínio validado. Caso uma fonte de renda seja incorporada, ela deve entrar no contrato como `float64`, não nula e `> 0`.
