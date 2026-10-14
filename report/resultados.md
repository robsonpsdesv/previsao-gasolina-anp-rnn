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
| **Etapa 1** — Previsão em nível de preço (seções 4 a 11) | GRU, MAE validação 0,446 | Melhor recorrente nessa formulação, mas generalização insatisfatória (MAE teste 1,317; R² teste -5,306) |
| **Etapa 2** — Previsão com log-retornos (seção 12) | SimpleRNN, MAE validação 0,0313 | Forte melhora com representação transformada (MAE teste 0,0400; R² teste 0,9806) |
| **Comparação final com persistência** (seções 13 e 14) | MAE teste de 0,0391 | Melhor MAE de teste entre as referências destacadas; as redes da Etapa 2 apenas se aproximam (diferença de ~0,001 R$/L, execução única) |

A seção 13 repete a análise em outros splits cronológicos para ambas as etapas.

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

**Preço da gasolina e log-retorno semanal por split (60/20/20)**

| Split | Média | Desvio | Mín. | Mediana | Máx. | Retorno: média | Retorno: desvio | Retorno: curtose | Retorno: autocorr. lag 1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Treino | 2,80 | 0,43 | 2,08 | 2,62 | 3,85 | 0,0009 | 0,0072 | 58,6 | 0,14 |
| Validação | 4,79 | 0,85 | 3,80 | 4,47 | 7,27 | 0,0027 | 0,0116 | 12,1 | 0,48 |
| Teste | 5,99 | 0,57 | 4,79 | 6,10 | 7,39 | -0,0004 | 0,0137 | 14,7 | 0,47 |

![Série e splits](../results/figures/eda_serie_e_splits.png)

![Distribuição do preço por split](../results/figures/eda_distribuicao_preco_por_split.png)

![Distribuição do log-retorno por split](../results/figures/eda_distribuicao_log_retorno_por_split.png)

![Histogramas das features](../results/figures/eda_histogramas_features.png)

![Autocorrelação do preço e do log-retorno](../results/figures/eda_autocorrelacao.png)

Observações:

- **Deslocamento de nível entre os splits**: as medianas do preço são 2,62, 4,47 e 6,10
  R$/L (treino, validação, teste), e as distribuições de treino e teste praticamente não
  se sobrepõem (máximo do treino 3,85; mínimo do teste 4,79).
- **Log-retorno semanal**: a média é próxima de zero nos três splits, mas o desvio
  aumenta (0,0072 → 0,0116 → 0,0137) e as caudas são pesadas (curtose em excesso entre
  12 e 59), com choques como -9,4% na semana de 2022-07-03 e +8,4% em 2022-03-13. Ou
  seja, a série de retornos tem escala mais estável que o nível, mas não é homogênea.
- **Autocorrelação**: o preço em nível é altamente persistente (0,997 na defasagem 1 e
  0,897 na defasagem 26), o que é compatível com a força do baseline de persistência. O
  log-retorno tem autocorrelação positiva até cerca de 9 semanas (0,375 na defasagem 1
  na série completa; 0,14, 0,48 e 0,47 por split), indicando alguma dependência de
  curto prazo que um modelo poderia explorar.
- **Features de contagem não estacionárias**: `gasolina_num_postos` cresce de 485 a
  12.701 postos pesquisados ao longo da série, o que sugere mudança na abrangência da
  pesquisa (observação descritiva; não foi investigada a causa).
- Todas as features têm assimetria positiva moderada (0,3 a 0,95).

## 3. Pré-processamento e protocolo experimental

- **Split cronológico** (sem embaralhamento), **60% treino / 20% validação / 20%
  teste**: treino 2004-05-09 a 2017-09-03 (691 semanas), validação 2017-09-10 a
  2022-03-27 (230 semanas), teste 2022-04-03 a 2026-08-30 (231 semanas), sem
  sobreposição temporal (verificado automaticamente: `max(train) < min(val)` e
  `max(val) < min(test)`). Proporções definidas em `src/config.py`
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
dias). A amostra é atribuída à partição pela **data-base `t`** (treino: `t` ≤ última
semana de treino, e assim por diante). Datas exatas, para todos os splits, janelas e
alvos, em [`results/fronteiras_splits.csv`](../results/fronteiras_splits.csv)
(gerado por `python -m src.eda`).

Split 60/20/20, alvo em nível:

| Janela | Partição | Amostras | Primeira entrada | Primeira base `t` | Primeiro alvo | Última base `t` | Último alvo |
|---:|---|---:|---|---|---|---|---|
| 4 | Treino | 688 | 2004-05-09 | 2004-05-30 | 2004-06-06 | 2017-09-03 | 2017-09-10 |
| 4 | Validação | 230 | 2017-08-20 | 2017-09-10 | 2017-09-17 | 2022-03-27 | 2022-04-03 |
| 4 | Teste | 231 | 2022-03-13 | 2022-04-03 | 2022-04-10 | 2026-08-30 | 2026-09-06 |
| 12 | Treino | 680 | 2004-05-09 | 2004-07-25 | 2004-08-01 | 2017-09-03 | 2017-09-10 |
| 12 | Validação | 230 | 2017-06-25 | 2017-09-10 | 2017-09-17 | 2022-03-27 | 2022-04-03 |
| 12 | Teste | 231 | 2022-01-16 | 2022-04-03 | 2022-04-10 | 2026-08-30 | 2026-09-06 |
| 26 | Treino | 666 | 2004-05-09 | 2004-10-31 | 2004-11-07 | 2017-09-03 | 2017-09-10 |
| 26 | Validação | 230 | 2017-03-19 | 2017-09-10 | 2017-09-17 | 2022-03-27 | 2022-04-03 |
| 26 | Teste | 231 | 2021-10-10 | 2022-04-03 | 2022-04-10 | 2026-08-30 | 2026-09-06 |

Datas-base das partições nos demais splits (o último alvo de cada partição é a semana
seguinte; o do teste é sempre 2026-09-06):

| Split | Treino | Validação | Teste |
|---|---|---|---|
| 60/20/20 | até 2017-09-03 | 2017-09-10 a 2022-03-27 | 2022-04-03 a 2026-08-30 |
| 70/15/15 | até 2019-11-17 | 2019-11-24 a 2023-04-30 | 2023-05-07 a 2026-08-30 |
| 75/15/10 | até 2021-02-21 | 2021-02-28 a 2024-06-09 | 2024-06-16 a 2026-08-30 |
| 80/10/10 | até 2022-03-27 | 2022-04-03 a 2024-06-09 | 2024-06-16 a 2026-08-30 |

Pontos a observar nas fronteiras:

- O **alvo da última amostra de treino é o preço da primeira semana de validação** (por
  exemplo, `t` = 2017-09-03 tem alvo em 2017-09-10), e o da última de validação é o da
  primeira semana de teste. As partições são disjuntas pela data-base, mas esse rótulo
  de fronteira pertence ao período seguinte.
- As **entradas das primeiras amostras de validação e teste incluem semanas da
  partição anterior** (por exemplo, com janela de 26 semanas, a validação começa a usar
  entradas de 2017-03-19 e o teste, de 2021-10-10). São dados passados em relação ao
  instante da previsão, não informação futura.
- Os **scalers são ajustados nas linhas com `t` até a última semana de treino**; como o
  alvo dessa linha é o preço da primeira semana de validação, esse único valor entra no
  ajuste do scaler do alvo (impacto esperado desprezível, mas registrado).
- No alvo em log-retorno, a primeira linha da série é descartada (não há diferença para
  ela); a primeira entrada do treino passa a ser 2004-05-16 e a primeira base `t` com
  janela de 4 semanas, 2004-06-06. Os limites de validação e teste coincidem com os do
  alvo em nível.

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
  learning_rate=0.00458, batch_size=64`, MAE de validação = 0.4456 R$/L (média das 20
  tentativas: 0.717). Registro completo em
  [`results/optuna_trials.csv`](../results/optuna_trials.csv) e
  [`results/optuna_best_params.json`](../results/optuna_best_params.json).

## 4. Etapa 1 — Resultados de validação em nível de preço e seleção dos modelos promissores

Tabela completa em [`results/experiments.csv`](../results/experiments.csv). Baseline
de persistência (referência):

| Split | MAE (R$/L) | RMSE (R$/L) | R² |
|---|---:|---:|---:|
| Treino | 0,0082 | 0,0212 | 0,9976 |
| Validação | 0,0356 | 0,0644 | 0,9944 |
| Teste | 0,0391 | 0,0818 | 0,9788 |

Comparação principal (3 janelas × 3 arquiteturas, unidades=64, 1 camada, dropout=0,2),
por MAE de validação:

| Janela | SimpleRNN | GRU | LSTM |
|---:|---:|---:|---:|
| 4 semanas | 0,856 | **0,754** | 0,883 |
| 12 semanas | 0,851 | 0,798 | 0,883 |
| 26 semanas | 0,893 | 0,827 | 0,909 |

A GRU na janela de 4 semanas foi a melhor combinação da comparação principal e foi
usada como base para a grade de regularização e para o Optuna. As grades de unidades
e profundidade foram executadas na janela de 12 semanas.

## 5. Três melhores configurações (por MAE de validação)

| # | Config. | Unid. | Camadas | Dropout | L2 | Parâmetros | Épocas | MAE val | RMSE val | R² val | MAE teste | RMSE teste | R² teste | Tempo (s) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | GRU (Optuna), janela 4 | 96 | 2 | 0,096 | 0,0 | 86.497 | 16 | **0,446** | 0,784 | 0,175 | 1,317 | 1,411 | -5,306 | 3,8 |
| 2 | GRU, janela 12 | 128 | 1 | 0,2 | 0,0 | 53.121 | 22 | 0,485 | 0,760 | 0,225 | 1,239 | 1,324 | -4,553 | 4,4 |
| 3 | GRU (L2), janela 4 | 64 | 1 | 0,0 | 0,001 | 14.273 | 100 | 0,624 | 0,896 | -0,077 | 1,460 | 1,538 | -6,487 | 7,5 |

As três melhores configurações são todas **GRU**. A 4ª colocada por MAE de validação
foi LSTM com 128 unidades (0,6249), seguida por SimpleRNN com 128 unidades (0,6257) —
praticamente empatadas com a 3ª colocada (0,6237). Modelos e figuras das três configurações estão em
`results/models/top{1,2,3}_*.keras` e `results/figures/*_top{1,2,3}_*.png`.

Gráficos (curvas de treino/validação, real×previsto e resíduos) em `results/figures/`:
`*_top{1,2,3}_*.png` (três melhores configurações), `*_optuna_best_GRU_w4_u96_L2.png`
e as três arquiteturas na janela de 12 semanas
(`*_principal_{SimpleRNN,GRU,LSTM}_w12_u64_L1.png`).

## 6. Avaliação final no conjunto de teste e custo × efetividade

**Nenhum modelo recorrente superou o baseline de persistência**, nem na validação nem
no teste: todos os R² de teste são negativos (entre -4,55 e -16,47), enquanto a
persistência atinge R²=0,979 no teste (MAE 0,039 R$/L, contra 1,24 R$/L do melhor
modelo recorrente). Mesmo na validação, apenas duas configurações têm R² positivo
(0,175 e 0,225). Foram feitas checagens de vazamento (ordem temporal dos splits, scalers ajustados só no treino, datas das fronteiras documentadas acima) e não se encontrou indício de vazamento nem de erro de implementação. A explicação mais consistente com as evidências é a **distribuição de preços diferente entre treino, validação e teste**:

| Split | Preço mínimo (R$/L) | Preço máximo (R$/L) | Média (R$/L) |
|---|---:|---:|---:|
| Treino (2004-05 a 2017-09) | 2,08 | 3,85 | 2,80 |
| Validação (2017-09 a 2022-03) | 3,80 | 7,27 | 4,79 |
| Teste (2022-04 a 2026-08) | 4,79 | 7,39 | 5,99 |

Com a divisão 60/20/20, o treino termina em setembro/2017 e cobre apenas preços até
R$ 3,85/L; o preço mínimo de validação (R$ 3,80) já é praticamente o máximo do treino
e o preço mínimo do teste (R$ 4,79) supera em ~R$ 0,94 o máximo do treino — reflexo da
forte alta de preços de combustíveis no Brasil a partir de 2018, e principalmente de
2021-2022. Uma interpretação plausível é que os modelos recorrentes aprendem uma função no espaço padronizado calibrado com as estatísticas do treino e extrapolam mal para faixas de preço nunca vistas, enquanto a persistência, por não depender de função ajustada, não é afetada pela mudança de nível. Essa é uma hipótese compatível com os dados, não testada isoladamente (por exemplo, por ablação ou por treino com preços deflacionados). 

Um indício adicional, compatível com essa interpretação: quase todos os modelos pararam o treinamento
cedo (16 a 24 épocas, com `patience=15`), ou seja, a melhor época da validação ocorreu
nas primeiras épocas (em geral entre a 1ª e a 9ª) e a validação piorou depois disso possivelmente à medida que o modelo se ajustava ao patamar de preços do treino. As exceções foram as
configurações com L2 (100 épocas) e sem regularização (59 épocas) na grade de
regularização.

**Custo computacional**: o pior caso (GRU/LSTM com 2 camadas, janela 12) treinou em
menos de 6 segundos nesta máquina (Apple Silicon, CPU — `gpu_devices: []`), pois o
dataset é pequeno (~680 sequências de treino). Na comparação principal, tempo médio de
treino/parâmetros: SimpleRNN 2,1 s / 4.737, GRU 3,0 s / 14.273, LSTM 2,8 s / 18.753; o
tempo médio por época foi de 0,12 s, 0,18 s e 0,16 s, respectivamente. O custo cresce
também com a janela (média das três arquiteturas: 2,2 s com 4 semanas, 2,6 s com 12
semanas e 3,1 s com 26 semanas).

## 7. SimpleRNN × GRU × LSTM (análise com base nos experimentos reais)

- **SimpleRNN**: a arquitetura mais leve (4.737 parâmetros com 64 unidades) e a mais
  rápida (2,1 s em média, 0,12 s/época). Desempenho intermediário na comparação
  principal (MAE val 0,851-0,893), melhor que a LSTM em todas as janelas e pior que a
  GRU. Com 128 unidades chegou a 0,626, quase empatada com a LSTM de mesma capacidade
  (0,625), mas com ~4× menos parâmetros (17.665 contra 70.273). Neste experimento não
  se pode atribuir a diferença em relação à GRU especificamente a dependências longas:
  janelas maiores não melhoraram nenhuma arquitetura (seção 8).
- **GRU**: foi a arquitetura com melhor desempenho de validação neste experimento —
  melhor das três em todas as janelas da comparação principal (0,754 / 0,798 / 0,827),
  dona das três melhores configurações gerais e do melhor resultado de teste entre os
  recorrentes (MAE 1,239 R$/L, GRU 128 unidades, janela 12). O custo adicional sobre a
  SimpleRNN foi moderado (~3× parâmetros, ~1,4× o tempo médio) e menor que o da LSTM
  em parâmetros (14.273 contra 18.753 com 64 unidades).
- **LSTM**: com 64 unidades teve o **pior** MAE de validação em todas as janelas da
  comparação principal (0,883 / 0,883 / 0,909) e o pior R² de teste, apesar de ter mais
  parâmetros e custo que SimpleRNN e GRU. A capacidade extra só mostrou valor com 128
  unidades (MAE val 0,625, 70.273 parâmetros), e mesmo assim não superou a GRU de 128
  unidades (0,485). Neste dataset e com a divisão 60/20/20, a LSTM não justificou seu
  custo adicional.
- Os conjuntos de unidades mostraram um padrão consistente: em todas as três
  arquiteturas, MAE de validação diminuiu monotonicamente de 32 para 64 e para 128
  unidades (SimpleRNN 0,968 → 0,851 → 0,626; GRU 0,844 → 0,798 → 0,485; LSTM 1,048 →
  0,883 → 0,625).

## 8. Janelas temporais

- **4 semanas**: foi suficiente e **a melhor janela** na comparação principal para
  GRU (0,754) e LSTM (0,883, empatada com 12 semanas); para SimpleRNN, 12 semanas foi
  ligeiramente melhor (0,851 contra 0,856).
- **12 semanas**: não trouxe melhora consistente frente a 4 semanas — piorou a GRU
  (0,798) e manteve a LSTM. Ainda assim, foi usada nas grades de unidades e profundidade;
  o melhor resultado entre elas (GRU, 128 unidades) é o 2º colocado geral.
- **26 semanas**: pior MAE de validação das três arquiteturas (0,893 / 0,827 / 0,909) e
  maior tempo de treino. Não houve evidência de que janelas longas ajudem a capturar
  dependências de mais longo prazo nesta série.
- **Conclusão sobre janelas**: a média do MAE de validação das três arquiteturas
  cresceu com a janela (0,831 com 4 semanas, 0,844 com 12 e 0,876 com 26), e o custo
  também (2,2 s, 2,6 s e 3,1 s). A janela de **4 semanas generalizou melhor** (menor
  erro e menor custo). Como apenas ~680 sequências de treino estão disponíveis e a
  variável-alvo já é uma feature em `t`, janelas maiores aumentam a dimensão de entrada
  sem informação adicional relevante para prever 1 semana à frente. A variação entre janelas (diferenças de até ~0,07 R$/L no MAE de validação) é pequena frente ao erro de teste em nível; a hipótese mais consistente para a baixa generalização é a mudança de patamar de preços (seção 6), e não o tamanho da janela.

## 9. Profundidade e regularização

- **Profundidade** (1 vs 2 camadas, janela 12, 64 unidades, dropout 0,2): efeito
  dependente da arquitetura. Na GRU, empilhar uma segunda camada **melhorou** o MAE de
  validação (0,798 → 0,698), ao custo de 2,7× mais parâmetros (39.233 contra 14.273) e
  quase o dobro do tempo (5,4 s contra 2,9 s). Na LSTM piorou (0,883 → 1,004; 51.777
  parâmetros, 5,9 s contra 2,8 s). Em ambos os casos o ganho/perda foi pequeno frente ao
  efeito de aumentar unidades em uma única camada (GRU de 128 unidades: 0,485).
- **Regularização** (GRU, janela 4, 64 unidades — melhor candidato da comparação
  principal): sem regularização MAE val 0,695 / teste 1,643 (59 épocas); dropout 0,2
  (comparação principal) 0,754 / 1,825; dropout 0,4 + recurrent dropout 0,2 0,735 /
  1,797; L2 (1e-3, sem dropout) **0,624 / 1,460** (100 épocas, ou seja, não houve
  parada antecipada). O L2 foi a técnica mais eficaz: reduziu o MAE de validação em
  ~10% frente a nenhuma regularização, enquanto o dropout (ruído estocástico) não
  trouxe ganho neste problema de séries pequenas e fora da faixa de treino.
- **Optuna**: a busca encontrou uma GRU com 2 camadas, 96 unidades e `batch_size=64`
  (MAE val 0,446), melhorando em ~41% o melhor da comparação principal (0,754) e em
  ~29% a melhor configuração da grade de regularização (0,624). Os três melhores
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
  seed=42; diferenças pequenas entre configurações (por exemplo, SimpleRNN-128 0,626 e
  LSTM-128 0,625) não devem ser interpretadas como significativas.
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
para 0,446 R$/L, resultado possivelmente associado à capacidade da GRU de controlar a
retenção e a atualização de informações temporais por meio de suas portas internas.

Entretanto, a avaliação no conjunto de teste revelou uma degradação significativa do
desempenho, com MAE de 1,317 R$/L e R² de -5,306. Além disso, o baseline de persistência
apresentou resultados substancialmente superiores. Portanto, embora a GRU tenha sido a
arquitetura neural mais promissora na etapa de validação, os resultados não sustentam
sua superioridade preditiva em relação ao baseline, sendo necessárias investigações
adicionais sobre generalização e metodologia de avaliação.

Detalhamento: sob um protocolo controlado (mesmos dados, split 60/20/20, normalização, otimizador,
loss e callbacks), a **GRU foi a arquitetura recorrente mais eficaz** para prever o
preço da gasolina em `t+1` neste recorte: ficou com as três melhores configurações, com
destaque para a busca Optuna (GRU de 2 camadas, 96 unidades, janela de 4 semanas) e para
GRU com 128 unidades. A SimpleRNN foi a mais leve e rápida, e a LSTM com 64 unidades foi
a menos eficaz, só se aproximando das demais com 128 unidades. Janelas curtas (4
semanas), mais unidades e regularização L2 foram os fatores que mais reduziram o erro de
validação. Nenhuma das três arquiteturas superou o baseline de persistência no conjunto
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
| Persistência | 0,0356 | 0,0391 | 0,0818 | 0,9788 |
| GRU w4 (nível, Optuna — seção 5) | 0,446 | 1,317 | 1,411 | -5,306 |
| SimpleRNN w4 (retorno, 64 un.) | 0,0323 | 0,0403 | 0,0774 | 0,9810 |
| GRU w4 (retorno, 64 un.) | 0,0331 | 0,0411 | 0,0815 | 0,9790 |
| LSTM w4 (retorno, 64 un.) | 0,0341 | 0,0409 | 0,0821 | 0,9786 |
| **SimpleRNN w4 (retorno, Optuna: 32 un., 1 camada)** | **0,0313** | 0,0400 | 0,0784 | 0,9806 |

- O MAE de teste caiu de 1,317 para ~0,040 R$/L (redução de ~97%) e o R² de teste foi de
  -5,306 para ~0,98, o que é consistente com a hipótese de que a falha anterior estava associada à representação (nível de preço não estacionário) e não apenas à capacidade das redes. Como features e alvo foram alterados em conjunto, o experimento não isola o efeito de cada mudança nem prova que essa seja a única causa.
- Frente à persistência, os modelos de retorno têm **MAE de validação menor** (0,0313
  contra 0,0356) e **RMSE de teste ligeiramente menor** (0,077-0,078 contra 0,082,
  SimpleRNN), mas **MAE de teste praticamente igual ou levemente pior** (0,040 contra
  0,0391). Portanto, as redes agora igualam o baseline, sem evidência de superioridade
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

**Cobertura de preços do treino em cada split** (validação e teste são sempre posteriores):

| Split | Treino até | Sem. treino/val/teste | Preço máx. treino | Preço mín.–máx. teste |
|---|---|---:|---:|---:|
| 60/20/20 | 2017-09-03 | 691 / 230 / 231 | 3,85 | 4,79–7,39 |
| 70/15/15 | 2019-11-17 | 806 / 172 / 174 | 4,73 | 5,21–6,78 |
| 75/15/10 | 2021-02-21 | 864 / 172 / 116 | 5,17 | 5,85–6,78 |
| 80/10/10 | 2022-03-27 | 921 / 115 / 116 | 7,27 | 5,85–6,78 |

**Alvo em nível** (melhor configuração por MAE de validação em cada split):

| Split | Melhor config. (val) | MAE val | MAE teste | R² teste | Persistência: MAE teste | Persistência: R² teste |
|---|---|---:|---:|---:|---:|---:|
| 60/20/20 | GRU, janela 4 | 0,754 | 1,825 | -10,57 | 0,0391 | 0,9788 |
| 70/15/15 | LSTM, janela 12 | 0,612 | 0,716 | -3,51 | 0,0256 | 0,9771 |
| 75/15/10 | LSTM, janela 12 | 0,547 | 0,535 | -6,27 | 0,0196 | 0,9634 |
| 80/10/10 | LSTM, janela 12 | 0,233 | 0,085 | 0,644 | 0,0196 | 0,9634 |

**Alvo em log-retorno** (melhor configuração por MAE de validação em cada split):

| Split | Melhor config. (val) | MAE val | MAE teste | RMSE teste | R² teste | Persistência: MAE val / MAE teste / RMSE teste |
|---|---|---:|---:|---:|---:|---|
| 60/20/20 | SimpleRNN, janela 4 | 0,0323 | 0,0403 | 0,0774 | 0,9810 | 0,0356 / 0,0391 / 0,0818 |
| 70/15/15 | GRU, janela 26 | 0,0485 | 0,0274 | 0,0523 | 0,9775 | 0,0583 / 0,0256 / 0,0528 |
| 75/15/10 | GRU, janela 12 | 0,0521 | 0,0228 | 0,0367 | 0,9672 | 0,0565 / 0,0196 / 0,0387 |
| 80/10/10 | LSTM, janela 4 | 0,0529 | 0,0239 | 0,0366 | 0,9672 | 0,0589 / 0,0196 / 0,0387 |

Observações:

- **Em nível, o erro de teste cai à medida que o treino cobre faixas de preço mais
  altas**: média das 9 configurações de 2,02 R$/L (60/20/20) para 1,02 (70/15/15),
  0,86 (75/15/10) e 0,15 (80/10/10). Só no 80/10/10, em que o treino já contém o pico de
  R$ 7,27 e todo o intervalo de preços do teste, o R² de teste fica positivo (0,64 na
  melhor configuração; as médias por arquitetura variam de -0,23 a 0,32). Mesmo assim o
  MAE (0,085 na melhor, 0,15 na média) é 4 a 8 vezes o da persistência (0,0196). Isso é consistente com a hipótese de que a falha em nível está ligada à extrapolação para faixas de preço não vistas, embora o experimento não isole essa causa de outras diferenças entre os períodos (volatilidade, regime de preços).
- **Em log-retorno, o erro é estável e baixo em todos os splits** (R² de teste de 0,97 a
  0,98), e a conclusão da seção 12 se mantém: o **MAE de validação é menor que o da
  persistência em todos os splits** (por exemplo 0,0529 contra 0,0589 no 80/10/10), o
  **RMSE de teste é igual ou menor** nas melhores configurações (SimpleRNN/GRU, p. ex.
  0,0348 contra 0,0387 no 80/10/10), mas o **MAE de teste é sempre pior que o da
  persistência** (mínimo/média entre as 9 configurações: 0,0400/0,0408 contra 0,0391 no
  60/20/20; 0,0270/0,0284 contra 0,0256 no 70/15/15; 0,0227/0,0238 contra 0,0196 no
  75/15/10; 0,0215/0,0231 contra 0,0196 no 80/10/10). Ou seja, as redes
  reduzem erros grandes (RMSE), mas não superam a persistência nos erros típicos (MAE).
- **Splits com teste menor são mais sensíveis**: nos splits 75/15/10 e 80/10/10 o teste
  tem 116 semanas (os dois compartilham o mesmo conjunto de teste, de modo que a
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

Conjunto de teste do split 60/20/20 (2022-04-03 a 2026-08-30), melhor modelo por MAE de
validação em cada etapa (ambos com busca Optuna) contra o baseline:

| Modelo | MAE val | MAE teste | RMSE teste | R² teste |
|---|---:|---:|---:|---:|
| Etapa 1 — GRU em nível (Optuna) | **0,446** | 1,317 | 1,411 | -5,306 |
| Etapa 2 — SimpleRNN em log-retorno (Optuna) | **0,0313** | 0,0400 | 0,0784 | 0,9806 |
| Persistência | 0,0356 | **0,0391** | 0,0818 | 0,9788 |

- **Etapa 1**: a GRU foi a melhor rede recorrente na validação, mas a avaliação em teste
  mostra generalização insatisfatória, associada, segundo a hipótese mais consistente com as evidências, à mudança de patamar de preços (seção 6); as checagens realizadas não indicaram vazamento nem erro de implementação.
- **Etapa 2**: a representação em log-retorno reduz o MAE de teste em ~97% e leva o R²
  de teste a ~0,98; a SimpleRNN passou a ser a melhor arquitetura na validação.
- **Persistência**: tem o **menor MAE de teste** (0,0391) entre as referências
  destacadas. A rede da Etapa 2 fica a ~0,001 R$/L, com RMSE e R² de teste marginalmente
  melhores; como cada configuração foi treinada uma vez (seed 42), não há evidência de
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
