# Previsão do Preço da Gasolina — Redes Recorrentes (SimpleRNN, GRU, LSTM)

Trabalho final da disciplina de Deep Learning (IFG). Frente escolhida: **3.2 — Redes
Recorrentes**. O objetivo é prever o **preço médio de revenda da gasolina comum na
semana seguinte** a partir da série histórica multivariada de preços de combustíveis
da ANP, comparando arquiteturas SimpleRNN, GRU e LSTM sob protocolo experimental
controlado.

## Objetivo

Modelar um problema de regressão de série temporal multivariada, um passo à frente
($X_{t-n+1..t} \rightarrow y_{t+1}$), avaliando o efeito da arquitetura recorrente,
do tamanho da janela temporal, da quantidade de unidades/camadas e da regularização
sobre desempenho, custo computacional e capacidade de generalização.

## Dataset

**Série Histórica de Preços de Combustíveis e de GLP — ANP**, agregado semanal em
nível **Brasil** (nacional).

- Fonte oficial: [gov.br/anp — Série histórica do levantamento de preços](https://www.gov.br/anp/pt-br/assuntos/precos-e-defesa-da-concorrencia/precos/precos-revenda-e-de-distribuicao-combustiveis/serie-historica-do-levantamento-de-precos)
- Portal de Dados Abertos: [dados.gov.br — Série Histórica de Preços de Combustíveis e de GLP](https://dados.gov.br/dados/conjuntos-dados/serie-historica-de-precos-de-combustiveis-e-de-glp)
- Arquivos utilizados (baixados diretamente para `datasets/anp_combustiveis/raw/`):
  - `semanal-brasil-2004-a-2012.xlsx` (09/05/2004 a 29/12/2012)
  - `semanal-brasil-desde-2013.xlsx` (30/12/2012 até a atualização mais recente)

### Localização do dataset

Todo o dataset (arquivos brutos, processados, intermediários, cache e metadados)
reside dentro do projeto, em (ignorado pelo git):

```text
datasets/anp_combustiveis/
├── raw/            # XLSX originais da ANP (imutáveis)
├── processed/       # série semanal nacional tratada (parquet)
├── intermediate/     # artefatos intermediários
├── metadata/
└── cache/
```

O caminho pode ser sobrescrito pela variável de ambiente `ANP_DATA_ROOT`.

## Problema de aprendizado

- **Variável alvo**: preço médio de revenda da gasolina comum em `t+1` (R$/litro).
- **Features (t)**: preço médio, mínimo, máximo, desvio padrão e coeficiente de
  variação da gasolina; número de postos pesquisados; preço médio do etanol
  hidratado; preço médio do óleo diesel.
- **Horizonte**: 1 semana à frente (one-step-ahead).
- **Janelas testadas**: 4, 12 e 26 semanas.
- **Split cronológico**: 60% treino / 20% validação / 20% teste (padrão), sem
  embaralhamento; splits alternativos 70/15/15, 75/15/10 e 80/10/10 também avaliados.
- **Normalização**: `StandardScaler` ajustado somente no treino.
- **Duas formulações do alvo**:
  - **Etapa 1 — nível de preço** (`target_mode="level"`): prevê o preço em `t+1`.
  - **Etapa 2 — log-retorno** (`target_mode="delta"`): features e alvo em log-retorno
    semanal; o preço é reconstruído como `y_t · exp(r̂)`.
  - Em ambas, as métricas (MAE, RMSE, R²) são em R$/L e comparadas à persistência.

## Estrutura do projeto

```text
projeto-anp/
├── README.md
├── requirements.txt
├── datasets/               # versionado (ver datasets/README.md)
│   ├── README.md           # criação/consulta do banco DuckDB e conexão no DBeaver
│   ├── init.sql            # cria as tabelas do DuckDB a partir do dataset
│   ├── anp.duckdb          # banco DuckDB gerado
│   └── anp_combustiveis/
│       ├── raw/            # XLSX originais da ANP
│       ├── processed/      # série semanal nacional (parquet)
│       ├── intermediate/
│       ├── metadata/
│       └── cache/
├── src/
│   ├── config.py          # caminhos, criação de diretórios, seeds
│   ├── data_loader.py      # download/leitura resiliente dos XLSX da ANP
│   ├── preprocessing.py    # pivot, alvo, checks, split cronológico, scalers
│   ├── sequences.py        # create_sequences(X, y, window_size, horizon)
│   ├── models.py           # SimpleRNN / GRU / LSTM comparáveis
│   ├── training.py         # preparação por janela (nível/log-retorno, splits) + treino
│   ├── evaluation.py       # métricas na escala original + gráficos
│   ├── run_experiments.py        # Etapa 1: grade completa em nível + Optuna
│   ├── run_experiments_delta.py  # Etapa 2: grade em log-retorno + Optuna
│   ├── run_experiments_splits.py # splits 60/20/20, 70/15/15, 75/15/10, 80/10/10
│   ├── eda.py                    # estatísticas, distribuições e datas das fronteiras
│   └── utils.py            # seeds e relatório do ambiente
├── notebooks/
│   └── analysis.ipynb
├── results/
│   ├── experiments.csv             # Etapa 1
│   ├── optuna_trials.csv
│   ├── optuna_best_params.json
│   ├── experiments_delta.csv       # Etapa 2
│   ├── optuna_delta_trials.csv
│   ├── optuna_delta_best_params.json
│   ├── experiments_splits.csv      # sensibilidade ao split
│   ├── eda_estatisticas_*.csv      # estatísticas descritivas
│   ├── fronteiras_splits.csv       # datas de entradas/alvos por partição
│   ├── tables/                     # capturas das tabelas do DuckDB (DBeaver)
│   ├── figures/
│   └── models/
└── report/
    └── resultados.md
```

## Instalação

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Execução

Todos os comandos são executados na raiz do projeto, com o ambiente virtual ativo. Todos
usam `seed=42`; cada configuração é treinada uma vez.

```bash
# 1. Baixa dados (se necessário) e constrói a série processada
python -m src.preprocessing

# 2. Análise exploratória: estatísticas, gráficos de distribuição e datas das
#    fronteiras dos splits (results/eda_*.csv, fronteiras_splits.csv, figures/eda_*.png)
python -m src.eda

# 3. Etapa 1 — previsão em nível de preço (split 60/20/20): baseline, janelas,
#    arquiteturas, unidades, profundidade, regularização e busca Optuna
python -m src.run_experiments            # -> results/experiments.csv

# 4. Etapa 2 — previsão com log-retornos (split 60/20/20): grade 3 janelas x 3
#    arquiteturas + Optuna, comparada à persistência
python -m src.run_experiments_delta      # -> results/experiments_delta.csv

# 5. Sensibilidade ao split (60/20/20, 70/15/15, 75/15/10, 80/10/10), nos dois alvos
python -m src.run_experiments_splits    # -> results/experiments_splits.csv
```

Para usar outro split ou alvo em código próprio:

```python
from src.training import train_experiment
result, art = train_experiment("GRU", window_size=12, target_mode="delta",
                               split_fracs=(0.70, 0.15))  # 70/15/15
```

Com `target_mode="delta"`, o artefato retornado contém `price_true_{val,test}` e
`price_pred_{val,test}` (R$/L); em `"level"`, as métricas usam `art["target_scaler"]`
(ver `src/run_experiments_splits.py`).

O notebook `notebooks/analysis.ipynb` reproduz o pipeline da Etapa 1 de forma
interativa (exploração dos dados, gráficos e discussão); a Etapa 2 e a sensibilidade ao
split são executadas pelos scripts acima.

## Resultados

Ver [report/resultados.md](report/resultados.md): resumo em duas etapas (nível de preço
e log-retorno), análise exploratória, datas das fronteiras dos splits, tabela de
experimentos, sensibilidade ao split, comparação final com a persistência e limitações,
com números extraídos dos CSVs em `results/`.
