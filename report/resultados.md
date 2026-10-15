# Resultados — Previsão do Preço da Gasolina com Redes Recorrentes (ANP)

> **Prof. Dr.**: _Sérgio Canuto_

> **Integrantes do grupo**: _Robson Pereira da Silva e Ivânio José da Rocha Júnior_

> **Link do repositório GitHub**: [_https://github.com/robsonpsdesv/previsao-gasolina-anp-rnn_](https://github.com/robsonpsdesv/previsao-gasolina-anp-rnn)

## 1. Problema e objetivo

Previsão, um passo à frente, do **preço médio de revenda da gasolina comum no Brasil
na semana seguinte** ($X_{t-n+1..t} \rightarrow y_{t+1}$), a partir da série semanal
multivariada de preços de combustíveis da ANP. Comparação controlada entre
**SimpleRNN, GRU e LSTM**, variando janela temporal, unidades, profundidade e
regularização, com busca de hiperparâmetros via Optuna.

## Resumo dos resultados em duas etapas

Os resultados estão organizados em duas etapas, com a comparação final contra a persistência
(split 60/20/20, métricas em R$/L):

| Etapa | Principal resultado | Interpretação |
|---|---|---|
| **Etapa 1** — Previsão em nível de preço (seções 4 a 11) | GRU, MAE validação 0,435 | Melhor recorrente nessa formulação, mas generalização insatisfatória (MAE teste 1,322; R² teste -5,265) |
| **Etapa 2** — Previsão com log-retornos (seção 12) | SimpleRNN, MAE validação 0,0313 | Forte melhora com representação transformada (MAE teste 0,0412; R² teste 0,9804) |
| **Comparação final com persistência** (seções 13 e 14) | MAE teste de 0,0390 | Melhor MAE de teste entre as referências destacadas; as redes da Etapa 2 apenas se aproximam (diferença de ~0,001 a 0,002 R$/L, execução única) |

A seção 13 repete a análise em outros splits cronológicos para ambas as etapas. Todos os números deste relatório foram obtidos **após a correção das fronteiras temporais** descrita na seção 3 (antes dela, o alvo da última linha de treino pertencia à primeira semana de validação).

## 2. Fonte e caracterização do dataset

- **Fonte**: ANP — Série Histórica do Levantamento de Preços, agregado semanal nacional
  (`semanal-brasil-2004-a-2012.xlsx` + `semanal-brasil-desde-2013.xlsx`), baixados
  diretamente para `datasets/anp_combustiveis/raw/`.
- **Período**: 2004-05-09 a 2026-09-06 (1152 semanas após a construção da série e do
  alvo em `t+1`).
- **Granularidade**: semanal, nível Brasil (nacional), evitando misturar estados/municípios.
- **Produtos usados**: gasolina comum, etanol hidratado, óleo diesel.
- **Features** (8, todas em `t`): preço médio, mínimo, máximo, desvio padrão e
  coeficiente de variação da gasolina; número de postos pesquisados; preço médio do
  etanol; preço médio do diesel.
- **Alvo**: preço médio de revenda da gasolina em `t+1` (R$/litro).
- **Qualidade dos dados**: sem valores ausentes ou duplicados após o pivot; 4 lacunas
  pontuais na periodicidade semanal (maior delas 63 dias, em outubro/2020) que não
  comprometem a construção das janelas, pois `create_sequences` opera sobre a ordem
  das observações disponíveis, não sobre datas de calendário.

### Análise exploratória

Estatísticas e gráficos gerados por `python -m src.eda`
([`results/eda_estatisticas_features.csv`](../results/eda_estatisticas_features.csv),
[`results/eda_estatisticas_por_split.csv`](../results/eda_estatisticas_por_split.csv) e
`results/figures/eda_*.png`). A curtose é a curtose em excesso (0 para a normal).

**Features (série completa, 1152 semanas)**

| Feature | Média | Desvio | Mín. | Mediana | Máx. | Assimetria |
|---|---:|---:|---:|---:|---:|---:|
| gasolina_preco_medio (R$/L) | 3,83 | 1,44 | 2,08 | 3,30 | 7,39 | 0,73 |
| gasolina_preco_minimo (R$/L) | 3,17 | 1,19 | 1,39 | 2,75 | 6,29 | 0,75 |
| gasolina_preco_maximo (R$/L) | 5,11 | 1,95 | 2,85 | 4,52 | 9,79 | 0,81 |
| gasolina_desvio_padrao (R$/L) | 0,24 | 0,09 | 0,135 | 0,198 | 0,535 | 0,62 |
| gasolina_coef_variacao | 0,062 | 0,006 | 0,049 | 0,062 | 0,083 | 0,40 |
| gasolina_num_postos | 7163 | 2776 | 485 | 7229 | 12701 | 0,32 |
| etanol_preco_medio (R$/L) | 2,59 | 1,07 | 1,20 | 2,20 | 5,54 | 0,69 |
| diesel_preco_medio (R$/L) | 3,34 | 1,65 | 1,44 | 2,81 | 7,57 | 0,95 |

**Preço da gasolina e log-retorno semanal por split (60/20/20, partições por data-base `t`)**

| Split | Média | Desvio | Mín. | Mediana | Máx. | Retorno: média | Retorno: desvio | Retorno: curtose | Retorno: autocorr. lag 1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Treino | 2,79 | 0,43 | 2,08 | 2,62 | 3,78 | 0,0009 | 0,0072 | 59,7 | 0,15 |
| Validação | 4,78 | 0,84 | 3,80 | 4,47 | 7,27 | 0,0028 | 0,0116 | 11,8 | 0,48 |
| Teste | 5,99 | 0,57 | 4,79 | 6,10 | 7,39 | -0,0004 | 0,0136 | 14,8 | 0,47 |

![Série e splits](../results/figures/eda_serie_e_splits.png)

![Distribuição do preço por split](../results/figures/eda_distribuicao_preco_por_split.png)

![Distribuição do log-retorno por split](../results/figures/eda_distribuicao_log_retorno_por_split.png)

![Histogramas das features](../results/figures/eda_histogramas_features.png)

![Autocorrelação do preço e do log-retorno](../results/figures/eda_autocorrelacao.png)

Observações:

- **Deslocamento de nível entre os splits**: as medianas do preço são 2,62, 4,47 e 6,10
  R$/L (treino, validação, teste), e as distribuições de treino e teste praticamente não
  se sobrepõem (máximo do treino 3,78; mínimo do teste 4,79).
- **Log-retorno semanal**: a média é próxima de zero nos três splits, mas o desvio
  aumenta (0,0072 → 0,0116 → 0,0136) e as caudas são pesadas (curtose em excesso entre 12 e 60), com choques como -9,4% na semana de 2022-07-03 e +8,4% em 2022-03-13. Ou
  seja, a série de retornos tem escala mais estável que o nível, mas não é homogênea.
- **Autocorrelação**: o preço em nível é altamente persistente (0,997 na defasagem 1 e
  0,897 na defasagem 26), o que é compatível com a força do baseline de persistência. O
  log-retorno tem autocorrelação positiva até cerca de 9 semanas (0,375 na defasagem 1
  na série completa; 0,15, 0,48 e 0,47 por split), indicando alguma dependência de
  curto prazo que um modelo poderia explorar.
- **Features de contagem não estacionárias**: `gasolina_num_postos` cresce de 485 a
  12.701 postos pesquisados ao longo da série, o que sugere mudança na abrangência da
  pesquisa (observação descritiva; não foi investigada a causa).
- Todas as features têm assimetria positiva moderada (0,3 a 0,95).

## 3. Pré-processamento e protocolo experimental

- **Split cronológico** (sem embaralhamento), **60% treino / 20% validação / 20%
  teste**, definido para que os **alvos** de cada partição fiquem no seu próprio
  período. Por data-base `t`: treino 2004-05-09 a 2017-08-27 (690 semanas), validação
  2017-09-03 a 2022-03-20 (230) e teste 2022-03-27 a 2026-08-30 (232); os alvos
  (`t+1`) correspondentes vão até 2017-09-03, de 2017-09-10 a 2022-03-27 e de
  2022-04-03 a 2026-09-06. Verificado automaticamente: `max(train) < min(val)`,
  `max(val) < min(test)` e, em `python -m src.eda`, que o último alvo de cada
  partição precede o primeiro alvo da seguinte, para todos os splits, janelas e alvos. Proporções definidas em `src/config.py`
  (`TRAIN_FRAC`, `VAL_FRAC`, `TEST_FRAC`).
- **Normalização**: `StandardScaler` ajustado **somente no treino** (features e alvo
  separadamente), escolhido em vez de `MinMaxScaler` porque os preços de combustível
  não têm limite superior conhecido a priori e os períodos de validação e teste, como
  discutido abaixo, ultrapassam a faixa observada no treino — a padronização por
  z-score não "trava" a escala nos extremos vistos no treino.
- **Janelas**: 4, 12 e 26 semanas, geradas por `create_sequences(X, y, window_size)`.
- **Protocolo comparável**: mesmo batch size (32), mesma loss (MSE) e métrica (MAE),
  mesmo otimizador (Adam), `EarlyStopping(patience=15, restore_best_weights=True)` +
  `ReduceLROnPlateau`, mesma seed (42) e `shuffle=False` em todos os treinos.
- **Baseline**: persistência ($\hat{y}_{t+1} = y_t$).

### Datas das entradas e dos alvos nas fronteiras dos conjuntos

Convenção: uma amostra com data-base `t` usa como entrada as `w` observações
consecutivas disponíveis `t-w+1, ..., t` e tem como alvo o preço da observação seguinte
(`t+1`, ou seja, a próxima semana disponível — a maior lacuna, em outubro/2020, tem 63
dias). Cada amostra é atribuída à partição pela data-base `t`, com as últimas linhas de
treino e de validação recuadas em um passo (`horizon`) em `chronological_split`, de modo
que o **alvo de toda amostra de uma partição cai no período de alvos dessa partição**.
Datas exatas, para todos os splits, janelas e alvos, em
[`results/fronteiras_splits.csv`](../results/fronteiras_splits.csv) (gerado por
`python -m src.eda`).

Split 60/20/20, alvo em nível:

| Janela | Partição | Amostras | Primeira entrada | Primeira base `t` | Primeiro alvo | Última base `t` | Último alvo |
|---:|---|---:|---|---|---|---|---|
| 4 | Treino | 687 | 2004-05-09 | 2004-05-30 | 2004-06-06 | 2017-08-27 | 2017-09-03 |
| 4 | Validação | 230 | 2017-08-13 | 2017-09-03 | 2017-09-10 | 2022-03-20 | 2022-03-27 |
| 4 | Teste | 232 | 2022-03-06 | 2022-03-27 | 2022-04-03 | 2026-08-30 | 2026-09-06 |
| 12 | Treino | 679 | 2004-05-09 | 2004-07-25 | 2004-08-01 | 2017-08-27 | 2017-09-03 |
| 12 | Validação | 230 | 2017-06-18 | 2017-09-03 | 2017-09-10 | 2022-03-20 | 2022-03-27 |
| 12 | Teste | 232 | 2022-01-09 | 2022-03-27 | 2022-04-03 | 2026-08-30 | 2026-09-06 |
| 26 | Treino | 665 | 2004-05-09 | 2004-10-31 | 2004-11-07 | 2017-08-27 | 2017-09-03 |
| 26 | Validação | 230 | 2017-03-12 | 2017-09-03 | 2017-09-10 | 2022-03-20 | 2022-03-27 |
| 26 | Teste | 232 | 2021-10-03 | 2022-03-27 | 2022-04-03 | 2026-08-30 | 2026-09-06 |

Datas-base das partições em cada split (o último alvo de cada partição é a semana
seguinte; o do teste é sempre 2026-09-06):

| Split | Treino (base `t`) | Validação (base `t`) | Teste (base `t`) |
|---|---|---|---|
| 60/20/20 | até 2017-08-27 | 2017-09-03 a 2022-03-20 | 2022-03-27 a 2026-08-30 |
| 70/15/15 | até 2019-11-10 | 2019-11-17 a 2023-04-23 | 2023-04-30 a 2026-08-30 |
| 75/15/10 | até 2021-02-14 | 2021-02-21 a 2024-06-02 | 2024-06-09 a 2026-08-30 |
| 80/10/10 | até 2022-03-20 | 2022-03-27 a 2024-06-02 | 2024-06-09 a 2026-08-30 |

Pontos a observar nas fronteiras:

- O **último alvo do treino (2017-09-03) é anterior ao primeiro alvo da validação
  (2017-09-10)**, e o último da validação (2022-03-27) é anterior ao primeiro do teste
  (2022-04-03). A semana de 2017-09-03 aparece apenas como último rótulo do treino e como
  primeira entrada da validação (dado passado, não informação futura).
- As **entradas das primeiras amostras de validação e teste incluem semanas da
  partição anterior** (por exemplo, com janela de 26 semanas, a validação começa a usar
  entradas de 2017-03-12 e o teste, de 2021-10-03), sempre anteriores ao instante da
  previsão.
- Os **scalers são ajustados apenas nas linhas de treino**, cujos alvos terminam em
  2017-09-03; nenhum valor do período de validação ou teste entra no ajuste.
- No alvo em log-retorno, a primeira linha da série é descartada (não há diferença para
  ela); a primeira entrada do treino passa a ser 2004-05-16 e a primeira base `t` com
  janela de 4 semanas, 2004-06-06. Os limites de validação e teste coincidem com os do
  alvo em nível.

### Correção de fronteira em relação à versão anterior dos experimentos

Uma revisão do código mostrou que, na versão anterior, a última linha de treino (data-base
2017-09-03) tinha como alvo o preço da primeira semana de validação (3,872 R$/L). Esse
valor entrava como rótulo de treino e no ajuste do `StandardScaler` do alvo (a média mudava
cerca de 0,0016 R$/L e o desvio, 0,0016 R$/L), e o mesmo ocorria na fronteira
validação/teste. O efeito numérico esperado era pequeno, mas a separação não era
estritamente temporal. A correção está em `chronological_split` (`src/preprocessing.py`) e
**todos os experimentos foram reexecutados** (`run_experiments`, `run_experiments_delta`,
`run_experiments_splits`, `eda`); as tabelas deste relatório usam apenas os novos
resultados. Comparação das métricas principais (split 60/20/20):

| Resultado | Antes | Depois |
|---|---:|---:|
| Persistência: MAE / RMSE de teste | 0,0391 / 0,0818 | 0,0390 / 0,0816 |
| Etapa 1, GRU Optuna: MAE val / MAE teste | 0,446 / 1,317 | 0,435 / 1,322 |
| Etapa 2, SimpleRNN Optuna: MAE val / MAE teste | 0,0313 / 0,0400 | 0,0313 / 0,0412 |
| Etapa 2, Optuna: configuração escolhida | 32 un., 1 camada | 64 un., 2 camadas |
| Etapa 1, 3ª melhor por MAE val | GRU + L2 (0,624) | LSTM 128 un. (0,611) |

As variações de métricas ficaram na 2ª–3ª casa decimal e as conclusões qualitativas não
mudaram; mudaram a 3ª colocada da Etapa 1 e a configuração escolhida pelo Optuna na
Etapa 2.

### Hiperparametrização (Optuna)

- **Arquitetura/janela de partida**: a melhor combinação da comparação principal
  (GRU, janela=4), usada como base para a busca.
- **Espaço de busca**: `units ∈ {16,32,64,96,128}` (capacidade), `n_layers ∈ {1,2}`
  (profundidade), `dropout ∈ [0, 0.5]` (regularização), `learning_rate ∈ [1e-4, 5e-3]`
  em escala log (velocidade/estabilidade de convergência), `batch_size ∈ {16,32,64}`
  (custo/estabilidade do gradiente). Intervalos escolhidos por serem os
  hiperparâmetros com maior potencial de alterar desempenho, generalização e custo
  computacional para redes recorrentes pequenas, conforme recomendado no enunciado.
- **Métrica otimizada**: MAE de validação (escala original, R$/L), calculada apenas
  no conjunto de validação — nunca no teste.
- **Amostrador**: TPE (`optuna.samplers.TPESampler`, seed=42), **20 tentativas**
  (`n_trials=20`), `max_epochs=60` e `patience=10` por tentativa (reduzidos frente aos
  100/15 usados na comparação principal para manter o custo da busca controlado).
- **Resultado**: melhor tentativa (trial 11) — `units=96, n_layers=2, dropout=0.096,
  learning_rate=0.00458, batch_size=64`, MAE de validação = 0.4351 R$/L (média das 20
  tentativas: 0.705). Registro completo em
  [`results/optuna_trials.csv`](../results/optuna_trials.csv) e
  [`results/optuna_best_params.json`](../results/optuna_best_params.json).

## 4. Etapa 1 — Resultados de validação em nível de preço e seleção dos modelos promissores

Tabela completa em [`results/experiments.csv`](../results/experiments.csv). Baseline
de persistência (referência; treino: MAE 0,0081, RMSE 0,0212, R² 0,9976):

| Split | MAE (R$/L) | RMSE (R$/L) | R² |
|---|---:|---:|---:|
| Validação | 0,0356 | 0,0644 | 0,9943 |
| Teste | 0,0390 | 0,0816 | 0,9792 |

Comparação principal (3 janelas × 3 arquiteturas, unidades=64, 1 camada, dropout=0,2),
por MAE de validação:

| Janela | SimpleRNN | GRU | LSTM |
|---:|---:|---:|---:|
| 4 semanas | 0,853 | **0,742** | 0,872 |
| 12 semanas | 0,849 | 0,788 | 0,874 |
| 26 semanas | 0,883 | 0,818 | 0,898 |

A GRU na janela de 4 semanas foi a melhor combinação da comparação principal e foi
usada como base para a grade de regularização e para o Optuna. As grades de unidades
e profundidade foram executadas na janela de 12 semanas.

## 5. Três melhores configurações (por MAE de validação)

| # | Config. | Unid. | Camadas | Dropout | L2 | Parâmetros | Épocas | MAE val | RMSE val | R² val | MAE teste | RMSE teste | R² teste | Tempo (s) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | GRU (Optuna), janela 4 | 96 | 2 | 0,096 | 0,0 | 86.497 | 16 | **0,435** | 0,768 | 0,186 | 1,322 | 1,417 | -5,265 | 3,9 |
| 2 | GRU, janela 12 | 128 | 1 | 0,2 | 0,0 | 53.121 | 22 | 0,473 | 0,747 | 0,230 | 1,246 | 1,333 | -4,541 | 4,4 |
| 3 | LSTM, janela 12 | 128 | 1 | 0,2 | 0,0 | 70.273 | 17 | 0,611 | 0,904 | -0,128 | 1,465 | 1,546 | -6,450 | 3,6 |

As duas primeiras configurações são **GRU**; a terceira é uma LSTM com 128 unidades. A 4ª
colocada por MAE de validação foi SimpleRNN com 128 unidades (0,622), seguida por GRU com L2
(0,669); a 3ª e a 4ª (0,611 e 0,622) são praticamente empatadas. Modelos e figuras das três configurações
estão em `results/models/top{1,2,3}_*.keras` e `results/figures/*_top{1,2,3}_*.png`.

Gráficos (curvas de treino/validação, real×previsto e resíduos) em `results/figures/`:
`*_top{1,2,3}_*.png` (três melhores configurações), `*_optuna_best_GRU_w4_u96_L2.png`
e as três arquiteturas na janela de 12 semanas
(`*_principal_{SimpleRNN,GRU,LSTM}_w12_u64_L1.png`).

## 6. Avaliação final no conjunto de teste e custo × efetividade

**Nenhum modelo recorrente superou o baseline de persistência**, nem na validação nem
no teste: todos os R² de teste são negativos (entre -4,54 e -16,82), enquanto a
persistência atinge R²=0,979 no teste (MAE 0,039 R$/L, contra 1,25 R$/L do melhor
modelo recorrente). Mesmo na validação, apenas duas configurações têm R² positivo
(0,230 e 0,186). Foram feitas checagens de vazamento (ordem temporal dos splits, alvos de cada
partição dentro do seu período, scalers ajustados só no treino, datas das fronteiras documentadas na seção 3) e não se
encontrou indício de vazamento nem de erro de implementação. A explicação mais consistente com as evidências é a
**distribuição de preços diferente entre treino, validação e teste** (partições por data-base):

| Split | Preço mínimo (R$/L) | Preço máximo (R$/L) | Média (R$/L) |
|---|---:|---:|---:|
| Treino (2004-05 a 2017-08) | 2,08 | 3,78 | 2,79 |
| Validação (2017-09 a 2022-03) | 3,80 | 7,27 | 4,78 |
| Teste (2022-03 a 2026-08) | 4,79 | 7,39 | 5,99 |

Com a divisão 60/20/20, o treino termina em agosto/2017 e cobre apenas preços até
R$ 3,78/L; o preço mínimo de validação (R$ 3,80) já é praticamente o máximo do treino
e o preço mínimo do teste (R$ 4,79) supera em ~R$ 1,01 o máximo do treino — reflexo da
forte alta de preços de combustíveis no Brasil a partir de 2018, e principalmente de
2021-2022. Uma interpretação plausível é que os modelos recorrentes aprendem uma função no espaço padronizado calibrado com as
estatísticas do treino e extrapolam mal para faixas de preço nunca vistas, enquanto a persistência, por não depender de função
ajustada, não é afetada pela mudança de nível. Essa é uma hipótese compatível com os dados, não testada isoladamente (por
exemplo, por ablação ou por treino com preços deflacionados).

Um indício adicional, compatível com essa interpretação: quase todos os modelos pararam o treinamento
cedo (16 a 24 épocas, com `patience=15`), ou seja, a melhor época da validação ocorreu
nas primeiras épocas (entre a 1ª e a 9ª) e a validação piorou depois disso, possivelmente à medida que o modelo se ajustava ao
patamar de preços do treino. As exceções foram as configurações sem regularização (59 épocas) e com L2
(61 épocas) na grade de regularização.

**Custo computacional**: o pior caso (GRU/LSTM com 2 camadas, janela 12) treinou em
menos de 7 segundos nesta máquina (Apple Silicon, CPU — `gpu_devices: []`), pois o
dataset é pequeno (~680 sequências de treino). Na comparação principal, tempo médio de
treino/parâmetros: SimpleRNN 2,1 s / 4.737, GRU 3,0 s / 14.273, LSTM 2,8 s / 18.753; o
tempo médio por época foi de 0,12 s, 0,18 s e 0,16 s, respectivamente. O custo cresce
também com a janela (média das três arquiteturas: 2,2 s com 4 semanas, 2,6 s com 12
semanas e 3,1 s com 26 semanas).

## 7. SimpleRNN × GRU × LSTM (análise com base nos experimentos reais)

- **SimpleRNN**: a arquitetura mais leve (4.737 parâmetros com 64 unidades) e a mais
  rápida (2,1 s em média, 0,12 s/época). Desempenho intermediário na comparação
  principal (MAE val 0,849-0,883), melhor que a LSTM em todas as janelas e pior que a
  GRU. Com 128 unidades chegou a 0,622, praticamente empatada com a LSTM de mesma capacidade
  (0,611), mas com ~4× menos parâmetros (17.665 contra 70.273). Neste experimento não
  se pode atribuir a diferença em relação à GRU especificamente a dependências longas:
  janelas maiores não melhoraram nenhuma arquitetura (seção 8).
- **GRU**: foi a arquitetura com melhor desempenho de validação neste experimento —
  melhor das três em todas as janelas da comparação principal (0,742 / 0,788 / 0,818),
  dona das duas melhores configurações gerais e do melhor resultado de teste entre os
  recorrentes (MAE 1,246 R$/L, GRU 128 unidades, janela 12). O custo adicional sobre a
  SimpleRNN foi moderado (~3× parâmetros, ~1,4× o tempo médio) e menor que o da LSTM
  em parâmetros (14.273 contra 18.753 com 64 unidades).
- **LSTM**: com 64 unidades teve o **pior** MAE de validação em todas as janelas da
  comparação principal (0,872 / 0,874 / 0,898) e o pior R² de teste, apesar de ter mais
  parâmetros e custo que SimpleRNN e GRU. A capacidade extra só mostrou valor com 128
  unidades (MAE val 0,611, 70.273 parâmetros, 3ª melhor configuração geral), e mesmo assim não superou a GRU de 128
  unidades (0,473). Neste dataset e com a divisão 60/20/20, a LSTM de 64 unidades não justificou seu
  custo adicional.
- Os conjuntos de unidades mostraram um padrão consistente: em todas as três
  arquiteturas, MAE de validação diminuiu monotonicamente de 32 para 64 e para 128
  unidades (SimpleRNN 0,963 → 0,849 → 0,622; GRU 0,835 → 0,788 → 0,473; LSTM 1,038 →
  0,874 → 0,611).

## 8. Janelas temporais

- **4 semanas**: foi suficiente e **a melhor janela** na comparação principal para
  GRU (0,742) e LSTM (0,872, praticamente empatada com 12 semanas, 0,874); para
  SimpleRNN, 12 semanas foi ligeiramente melhor (0,849 contra 0,853).
- **12 semanas**: não trouxe melhora consistente frente a 4 semanas — piorou a GRU
  (0,788) e manteve a LSTM. Ainda assim, foi usada nas grades de unidades e profundidade;
  o melhor resultado entre elas (GRU, 128 unidades) é o 2º colocado geral.
- **26 semanas**: pior MAE de validação das três arquiteturas (0,883 / 0,818 / 0,898) e
  maior tempo de treino. Não houve evidência de que janelas longas ajudem a capturar
  dependências de mais longo prazo nesta série.
- **Conclusão sobre janelas**: a média do MAE de validação das três arquiteturas
  cresceu com a janela (0,823 com 4 semanas, 0,837 com 12 e 0,866 com 26), e o custo
  também (2,2 s, 2,6 s e 3,1 s). A janela de **4 semanas generalizou melhor** (menor
  erro e menor custo). Como apenas ~680 sequências de treino estão disponíveis e a
  variável-alvo já é uma feature em `t`, janelas maiores aumentam a dimensão de entrada
  sem informação adicional relevante para prever 1 semana à frente. A variação entre janelas (diferenças de até ~0,07 R$/L no MAE de validação) é pequena frente ao erro de teste em nível; a hipótese mais consistente para a baixa generalização é a mudança de patamar de preços (seção 6), e não o tamanho da janela.

## 9. Profundidade e regularização

- **Profundidade** (1 vs 2 camadas, janela 12, 64 unidades, dropout 0,2): efeito
  dependente da arquitetura. Na GRU, empilhar uma segunda camada **melhorou** o MAE de
  validação (0,788 → 0,691), ao custo de 2,7× mais parâmetros (39.233 contra 14.273) e
  cerca do dobro do tempo (6,1 s contra 3,0 s). Na LSTM piorou (0,874 → 0,988; 51.777
  parâmetros, 5,7 s contra 2,9 s). Em ambos os casos o ganho/perda foi pequeno frente ao
  efeito de aumentar unidades em uma única camada (GRU de 128 unidades: 0,473).
- **Regularização** (GRU, janela 4, 64 unidades — melhor candidato da comparação
  principal): sem regularização MAE val 0,690 / teste 1,659 (59 épocas); dropout 0,2
  (comparação principal) 0,742 / 1,833; dropout 0,4 + recurrent dropout 0,2 0,724 /
  1,804; L2 (1e-3, sem dropout) **0,669 / 1,612** (61 épocas). O L2 foi a técnica com
  menor MAE de validação, mas a diferença para nenhuma regularização é pequena (~3%) e, com execução única, não deve ser
  tomada como significativa; o dropout (ruído estocástico) não
  trouxe ganho neste problema de séries pequenas e fora da faixa de treino.
- **Optuna**: a busca encontrou uma GRU com 2 camadas, 96 unidades e `batch_size=64`
  (MAE val 0,435), melhorando em ~41% o melhor da comparação principal (0,742) e em
  ~35% a melhor configuração da grade de regularização (0,669). Os três melhores
  trials usaram `batch_size=64` e `learning_rate` alto (2,4e-3 a 4,6e-3).
- Nenhuma configuração exibiu explosão de loss, NaN ou previsões constantes. O sinal de
  overfitting predominante é o da seção 6: melhor época de validação nas primeiras épocas
  e piora subsequente por ajuste ao patamar de preços do treino.

## 10. Limitações

- **Distribuição não estacionária**: como discutido na seção 6, o preço da gasolina
  mudou de patamar após o treino (2018-2022), e validação e teste ficaram em grande
  parte fora da faixa vista no treino. Isso limita a validade de qualquer modelo
  treinado com normalização fixa; em um cenário real, seria necessário re-treinar
  periodicamente ou modelar variações (diferenças/retornos) em vez do nível de preço.
- **Causa não isolada**: a associação entre a falha em nível e a mudança de patamar de preços é uma hipótese sustentada por evidências indiretas (faixas de preço por split, melhora com splits que cobrem preços mais altos e com log-retornos). Não houve ablação separando features de alvo, nem teste com outras transformações (por exemplo, preços deflacionados), então outras causas (mudanças de regime, de volatilidade e de abrangência da pesquisa) não podem ser descartadas.
- **Seleção de modelos prejudicada**: como a própria validação também está fora da faixa
  de treino, o MAE de validação é um critério ruidoso para escolher entre modelos que
  extrapolam mal; os rankings entre arquiteturas variaram entre os recortes 70/15/15 e
  60/20/20.
- **Dataset pequeno para deep learning**: apesar de cobrir 22 anos, a granularidade
  semanal produz apenas ~1150 observações (~680 sequências de treino).
- **Execução única por configuração**: cada configuração foi treinada uma vez com
  seed=42; diferenças pequenas entre configurações (por exemplo, SimpleRNN-128 0,622 e
  LSTM-128 0,611) não devem ser interpretadas como significativas.
- **Optuna não explorou L2 nem a escolha de arquitetura**: a busca variou dropout e
  não L2, e foi feita apenas para a arquitetura/janela vencedora da comparação
  principal (GRU, janela 4).
- **Baseline forte**: para preços com alta autocorrelação semana-a-semana, a
  persistência é um adversário muito difícil de superar com poucos dados e a variável
  de preço entre as próprias features — um resultado consistente com a literatura de
  séries financeiras.

## 11. Conclusão da Etapa 1 (previsão em nível de preço)

A arquitetura GRU apresentou o melhor desempenho relativo entre as redes recorrentes
avaliadas, obtendo os menores valores de MAE de validação nas três janelas temporais
investigadas. A otimização de hiperparâmetros com Optuna reduziu o MAE de validação
para 0,435 R$/L, resultado possivelmente associado à capacidade da GRU de controlar a
retenção e a atualização de informações temporais por meio de suas portas internas.

Entretanto, a avaliação no conjunto de teste revelou uma degradação significativa do
desempenho, com MAE de 1,322 R$/L e R² de -5,265. Além disso, o baseline de persistência
apresentou resultados substancialmente superiores. Portanto, embora a GRU tenha sido a
arquitetura neural mais promissora na etapa de validação, os resultados não sustentam
sua superioridade preditiva em relação ao baseline, sendo necessárias investigações
adicionais sobre generalização e metodologia de avaliação.

Detalhamento: sob um protocolo controlado (mesmos dados, split 60/20/20, normalização, otimizador,
loss e callbacks), a **GRU foi a arquitetura recorrente mais eficaz** para prever o
preço da gasolina em `t+1` neste recorte: ficou com as duas melhores configurações, com
destaque para a busca Optuna (GRU de 2 camadas, 96 unidades, janela de 4 semanas) e para
GRU com 128 unidades; a terceira melhor foi uma LSTM com 128 unidades. A SimpleRNN foi a mais leve e rápida, e a LSTM com 64 unidades foi
a menos eficaz, só se aproximando das demais com 128 unidades. Janelas curtas (4
semanas) e mais unidades foram os fatores que mais reduziram o erro de
validação; o L2 teve efeito pequeno. Nenhuma das três arquiteturas superou o baseline de persistência no conjunto
de teste (nem na validação), resultado que as evidências disponíveis associam a uma mudança mensurável de patamar de preços entre os períodos de treino e validação/teste (hipótese não testada isoladamente); as checagens realizadas não indicaram vazamento de dados nem erro de implementação. Todos os números deste relatório
foram extraídos de `results/experiments.csv`, `results/optuna_trials.csv` e do log da
execução de `python -m src.run_experiments`.

## 12. Etapa 2 — Reformulação com log-retornos (tratamento da não estacionariedade)

Como a hipótese mais consistente com as evidências para a degradação no teste é a mudança de patamar de preços (seção 6), o problema foi reformulado: features e alvo passaram a ser **log-retornos semanais**
(`log(x_t / x_{t-1})` nas features; alvo `r = log(y_{t+1} / y_t)`), e o preço previsto é
reconstruído como `y_t · exp(r̂)`. A persistência equivale a `r̂ = 0`. Mesmo split
60/20/20, escalonamento ajustado só no treino, mesma grade (3 janelas × 3 arquiteturas,
64 unidades, dropout 0,2) e mesma busca Optuna (20 tentativas), com métricas em R$/L.
Código em `src/run_experiments_delta.py`; resultados em
[`results/experiments_delta.csv`](../results/experiments_delta.csv),
`results/optuna_delta_trials.csv` e `results/optuna_delta_best_params.json`.

| Modelo | MAE val | MAE teste | RMSE teste | R² teste |
|---|---:|---:|---:|---:|
| Persistência | 0,0356 | 0,0390 | 0,0816 | 0,9792 |
| GRU w4 (nível, Optuna — seção 5) | 0,435 | 1,322 | 1,417 | -5,265 |
| SimpleRNN w4 (retorno, 64 un.) | 0,0323 | 0,0404 | 0,0774 | 0,9813 |
| GRU w4 (retorno, 64 un.) | 0,0332 | 0,0411 | 0,0814 | 0,9794 |
| LSTM w4 (retorno, 64 un.) | 0,0342 | 0,0409 | 0,0820 | 0,9790 |
| **SimpleRNN w4 (retorno, Optuna: 64 un., 2 camadas)** | **0,0313** | 0,0412 | 0,0793 | 0,9804 |

- O MAE de teste caiu de 1,322 para ~0,041 R$/L (redução de ~97%) e o R² de teste foi de
  -5,265 para ~0,98, o que é consistente com a hipótese de que a falha anterior estava associada à representação (nível de preço não estacionário) e não apenas à capacidade das redes. Como features e alvo foram alterados em conjunto, o experimento não isola o efeito de cada mudança nem prova que essa seja a única causa.
- Frente à persistência, os modelos de retorno têm **MAE de validação menor** (0,0313
  contra 0,0356) e **RMSE de teste ligeiramente menor** (0,077-0,079 contra 0,082 nos
  modelos SimpleRNN), mas **MAE de teste praticamente igual ou levemente pior** (0,0398 a 0,0412 contra
  0,0390). Portanto, as redes agora igualam o baseline, sem evidência de superioridade
  clara; diferenças dessa magnitude vêm de uma única execução (seed 42) e não devem ser
  tratadas como significativas.
- Com a representação estacionária, o ranking entre arquiteturas mudou: a SimpleRNN
  ficou na frente da GRU e da LSTM (e foi escolhida pelo Optuna), de modo que a
  superioridade da GRU observada na seção 4 está associada ao regime de extrapolação em
  nível, e não a uma vantagem geral da arquitetura.

## 13. Sensibilidade ao split cronológico (Etapas 1 e 2; 70/15/15, 75/15/10, 80/10/10)

Para verificar se o problema de generalização depende da divisão temporal, foram
repetidos os experimentos com três splits adicionais, comparados ao 60/20/20 original.
Em cada split: 3 arquiteturas × 3 janelas (64 unidades, 1 camada, dropout 0,2, seed 42),
nos dois alvos (preço em nível e log-retorno, seção 12), mais a persistência calculada
nas mesmas amostras de validação/teste. Todas as métricas em R$/L. Código em
`src/run_experiments_splits.py`; resultados completos em
[`results/experiments_splits.csv`](../results/experiments_splits.csv).

**Cobertura de preços do treino em cada split** (validação e teste são sempre posteriores; semanas por data-base):

| Split | Treino até (base) | Sem. treino/val/teste | Preço máx. treino | Preço mín.–máx. teste |
|---|---|---:|---:|---:|
| 60/20/20 | 2017-08-27 | 690 / 230 / 232 | 3,78 | 4,79–7,39 |
| 70/15/15 | 2019-11-10 | 805 / 172 / 175 | 4,72 | 5,21–6,78 |
| 75/15/10 | 2021-02-14 | 863 / 172 / 117 | 4,92 | 5,85–6,78 |
| 80/10/10 | 2022-03-20 | 920 / 115 / 117 | 7,27 | 5,85–6,78 |

**Alvo em nível** (melhor configuração por MAE de validação em cada split):

| Split | Melhor config. (val) | MAE val | MAE teste | R² teste | Persistência: MAE teste | Persistência: R² teste |
|---|---|---:|---:|---:|---:|---:|
| 60/20/20 | GRU, janela 4 | 0,742 | 1,833 | -10,51 | 0,0390 | 0,9792 |
| 70/15/15 | LSTM, janela 12 | 0,592 | 0,659 | -2,91 | 0,0257 | 0,9775 |
| 75/15/10 | LSTM, janela 26 | 0,562 | 0,702 | -11,21 | 0,0195 | 0,9648 |
| 80/10/10 | LSTM, janela 12 | 0,238 | 0,085 | 0,65 | 0,0195 | 0,9648 |

**Alvo em log-retorno** (melhor configuração por MAE de validação em cada split):

| Split | Melhor config. (val) | MAE val | MAE teste | RMSE teste | R² teste | Persistência: MAE val / MAE teste / RMSE teste |
|---|---|---:|---:|---:|---:|---|
| 60/20/20 | SimpleRNN, janela 4 | 0,0323 | 0,0404 | 0,0774 | 0,9813 | 0,0356 / 0,0390 / 0,0816 |
| 70/15/15 | GRU, janela 26 | 0,0483 | 0,0275 | 0,0523 | 0,9778 | 0,0582 / 0,0257 / 0,0527 |
| 75/15/10 | LSTM, janela 4 | 0,0522 | 0,0237 | 0,0369 | 0,9678 | 0,0571 / 0,0195 / 0,0386 |
| 80/10/10 | LSTM, janela 4 | 0,0529 | 0,0237 | 0,0364 | 0,9686 | 0,0589 / 0,0195 / 0,0386 |

Observações:

- **Em nível, o erro de teste cai à medida que o treino cobre faixas de preço mais
  altas**: média das 9 configurações de 2,03 R$/L (60/20/20) para 1,01 (70/15/15),
  0,93 (75/15/10) e 0,13 (80/10/10). Só no 80/10/10, em que o treino já contém o pico de
  R$ 7,27 e todo o intervalo de preços do teste, o R² de teste fica positivo (0,65 na
  melhor configuração; as médias por arquitetura variam de 0,23 a 0,48). Mesmo assim o
  MAE (0,085 na melhor, 0,13 na média) é 4 a 7 vezes o da persistência (0,0195). Isso é consistente com a hipótese de que a falha em nível está ligada à extrapolação para faixas de preço não vistas, embora o experimento não isole essa causa de outras diferenças entre os períodos (volatilidade, regime de preços).
- **Em log-retorno, o erro é estável e baixo em todos os splits** (R² de teste de 0,96 a
  0,98), e a conclusão da seção 12 se mantém: o **MAE de validação é menor que o da
  persistência na melhor configuração de todos os splits** (por exemplo 0,0529 contra 0,0589 no 80/10/10; em média
  das 9 configurações, também em todos), o
  **RMSE de teste é menor que o da persistência** em 5, 9, 8 e 9 das 9 configurações
  (60/20/20, 70/15/15, 75/15/10 e 80/10/10; p. ex. média da GRU de 0,0346 contra 0,0386 no 80/10/10), mas o **MAE de teste é sempre pior que o da
  persistência** (mínimo/média entre as 9 configurações: 0,0398/0,0408 contra 0,0390 no 60/20/20; 0,0270/0,0285 contra 0,0257 no 70/15/15; 0,0227/0,0237 contra 0,0195 no 75/15/10; 0,0214/0,0229 contra 0,0195 no 80/10/10). Ou seja, as redes
  reduzem erros grandes (RMSE) em parte dos casos, mas não superam a persistência nos erros típicos (MAE).
- **Splits com teste menor são mais sensíveis**: nos splits 75/15/10 e 80/10/10 o teste
  tem 117 semanas (os dois compartilham o mesmo conjunto de teste, de modo que a
  persistência é idêntica), e as diferenças entre modelos (≈0,001-0,002 R$/L) são
  pequenas diante da variabilidade esperada. Os MAE e RMSE absolutos não são comparáveis
  entre splits, pois os períodos de teste diferem; a comparação válida é sempre contra a
  persistência no mesmo split.
- **A melhor arquitetura varia conforme o split e o alvo** (GRU, LSTM e SimpleRNN
  aparecem como melhores por MAE de validação), com diferenças pequenas e execução única
  por configuração; não há base para apontar uma arquitetura como superior de forma
  geral.
- **Conclusão da seção**: o split mais favorável para o alvo em nível é o 80/10/10, mas
  isso ocorre por incluir no treino o patamar de preços recente; a reformulação em log-retorno manteve desempenho estável em todos os splits avaliados. Em nenhum cenário as redes
  superaram claramente o baseline de persistência no conjunto de teste.

## 14. Comparação final com a persistência e conclusão geral

Conjunto de teste do split 60/20/20 (alvos de 2022-04-03 a 2026-09-06, 232 amostras), melhor modelo por MAE de
validação em cada etapa (ambos com busca Optuna) contra o baseline:

| Modelo | MAE val | MAE teste | RMSE teste | R² teste |
|---|---:|---:|---:|---:|
| Etapa 1 — GRU em nível (Optuna) | **0,435** | 1,322 | 1,417 | -5,265 |
| Etapa 2 — SimpleRNN em log-retorno (Optuna) | **0,0313** | 0,0412 | 0,0793 | 0,9804 |
| Persistência | 0,0356 | **0,0390** | 0,0816 | 0,9792 |

- **Etapa 1**: a GRU foi a melhor rede recorrente na validação, mas a avaliação em teste
  mostra generalização insatisfatória, associada, segundo a hipótese mais consistente com as evidências, à mudança de patamar de preços (seção 6); as checagens realizadas não indicaram vazamento nem erro de implementação.
- **Etapa 2**: a representação em log-retorno reduz o MAE de teste em ~97% e leva o R²
  de teste a ~0,98; a SimpleRNN passou a ser a melhor arquitetura na validação.
- **Persistência**: tem o **menor MAE de teste** (0,0390) entre as referências
  destacadas. A rede da Etapa 2 escolhida pelo Optuna fica a ~0,002 R$/L (0,0412) e a melhor configuração de teste da Etapa 2 a ~0,001 R$/L
  (0,0398); o modelo escolhido pelo Optuna tem RMSE e R² de teste marginalmente melhores que os da persistência. Como cada configuração foi treinada uma vez (seed 42), não há evidência de
  superioridade de nenhum dos dois. Nos demais splits (seção 13) a persistência também
  mantém o menor MAE de teste.
- **Conclusão geral**: a maior melhora observada coincidiu com a mudança de representação (log-retornos) e foi muito maior que as diferenças entre arquiteturas; como a mudança foi feita em conjunto (features e alvo), o ganho é atribuído à reformulação como um todo, sem isolar cada componente. Com a representação adequada, as redes recorrentes
  igualam, mas não superam de forma clara, o baseline de persistência.

## Anexo — Tabelas do dataset no DuckDB (visualização no DBeaver)

Complemento visual das tabelas do banco `datasets/anp.duckdb`, criadas por
[`datasets/init.sql`](../datasets/init.sql). As capturas mostram as primeiras 200 linhas
de cada tabela.

**`precos_semanal_raw`** — importação direta das duas planilhas da ANP
(`semanal-brasil-2004-a-2012.xlsx` e `semanal-brasil-desde-2013.xlsx`), sem remoção de
duplicatas, com todos os produtos e a coluna `arquivo_origem`.

![Tabela precos_semanal_raw no DBeaver](../results/tables/precos_semanal_raw.png)

**`precos_semanal`** — mesma estrutura, sem as semanas repetidas na fronteira entre os
dois arquivos, ordenada por `data_inicial` e `produto`.

![Tabela precos_semanal no DBeaver](../results/tables/precos_semanal.png)

**`serie_semanal_nacional`** — série processada usada nos experimentos: as 8 features
(gasolina, etanol e diesel) e o alvo `target_gasolina_t_plus_1`, uma linha por semana
(1152 semanas).

![Tabela serie_semanal_nacional no DBeaver](../results/tables/serie_semanal_nacional.png)
