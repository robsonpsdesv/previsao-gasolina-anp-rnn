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
- **Split cronológico**: 60% treino / 20% validação / 20% teste, sem embaralhamento.
- **Normalização**: `StandardScaler` ajustado somente no treino.

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
│   ├── training.py         # preparação por janela + rotina de treino
│   ├── evaluation.py       # métricas na escala original + gráficos
│   ├── run_experiments.py  # grade completa de experimentos + Optuna
│   └── utils.py            # seeds e relatório do ambiente
├── notebooks/
│   └── analysis.ipynb
├── results/
│   ├── experiments.csv
│   ├── optuna_trials.csv
│   ├── optuna_best_params.json
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

```bash
# 1. Baixa dados (se necessário) e constrói a série processada
python -m src.preprocessing

# 2. Roda toda a grade de experimentos (baseline, janelas, arquiteturas,
#    unidades, profundidade, regularização e busca Optuna)
python -m src.run_experiments
```

O notebook `notebooks/analysis.ipynb` reproduz o pipeline de forma interativa,
com as exploração dos dados, gráficos e discussão dos resultados.

## Resultados

Ver [report/resultados.md](report/resultados.md) para a tabela de experimentos,
gráficos e análise crítica (SimpleRNN x GRU x LSTM, efeito da janela temporal,
custo computacional e escolha das três melhores configurações), com números
extraídos diretamente de `results/experiments.csv`.
