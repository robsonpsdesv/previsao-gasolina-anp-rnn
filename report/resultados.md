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

## 4. Resultados de validação e seleção dos modelos promissores

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
(0,175 e 0,225). Isso não é um sinal de vazamento de dados nem de erro de
implementação — foi investigado (seção 40 do enunciado) e a causa identificada é a
**distribuição de preços diferente entre treino, validação e teste**:

| Split | Preço mínimo (R$/L) | Preço máximo (R$/L) | Média (R$/L) |
|---|---:|---:|---:|
| Treino (2004-05 a 2017-09) | 2,08 | 3,85 | 2,80 |
| Validação (2017-09 a 2022-03) | 3,80 | 7,27 | 4,79 |
| Teste (2022-04 a 2026-08) | 4,79 | 7,39 | 5,99 |

Com a divisão 60/20/20, o treino termina em setembro/2017 e cobre apenas preços até
R$ 3,85/L; o preço mínimo de validação (R$ 3,80) já é praticamente o máximo do treino
e o preço mínimo do teste (R$ 4,79) supera em ~R$ 0,94 o máximo do treino — reflexo da
forte alta de preços de combustíveis no Brasil a partir de 2018, e principalmente de
2021-2022. Os modelos recorrentes aprendem uma função no espaço padronizado calibrado
com as estatísticas do treino e extrapolam mal para faixas de preço nunca vistas; a
persistência, por não depender de nenhuma função ajustada, é imune a essa mudança de
nível. 

Um indício adicional do mesmo problema: quase todos os modelos pararam o treinamento
cedo (16 a 24 épocas, com `patience=15`), ou seja, a melhor época da validação ocorreu
nas primeiras épocas (em geral entre a 1ª e a 9ª) e a validação piorou depois disso à
medida que o modelo se ajustava ao patamar de preços do treino. As exceções foram as
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
  sem informação adicional relevante para prever 1 semana à frente. O fator dominante
  da baixa generalização, de todo modo, é a mudança de patamar de preços (seção 6), e
  não o tamanho da janela.

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

## 11. Conclusão

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
de teste (nem na validação), resultado explicado por uma mudança real e mensurável de
patamar de preços entre os períodos de treino e validação/teste — não por vazamento de
dados, erro de implementação ou métricas fabricadas. Todos os números deste relatório
foram extraídos de `results/experiments.csv`, `results/optuna_trials.csv` e do log da
execução de `python -m src.run_experiments`.

## 12. Reformulação com log-retornos (tratamento da não estacionariedade)

Como a causa da degradação no teste é a mudança de patamar de preços (seção 6), o
problema foi reformulado: features e alvo passaram a ser **log-retornos semanais**
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
  -5,306 para ~0,98, confirmando que a falha anterior era de representação (nível de
  preço não estacionário), não de capacidade das redes.
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
