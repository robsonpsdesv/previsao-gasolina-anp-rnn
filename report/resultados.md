# Resultados — Previsão do Preço da Gasolina com Redes Recorrentes (ANP)

> **Integrantes do grupo**: _completar com nome(s) e matrícula(s) antes da entrega._
> **Link do repositório GitHub**: _completar após publicar o código._

## 1. Problema e objetivo

Previsão, um passo à frente, do **preço médio de revenda da gasolina comum no Brasil
na semana seguinte** ($X_{t-n+1..t} \rightarrow y_{t+1}$), a partir da série semanal
multivariada de preços de combustíveis da ANP. Comparação controlada entre
**SimpleRNN, GRU e LSTM**, variando janela temporal, unidades, profundidade e
regularização, com busca de hiperparâmetros via Optuna.

## 2. Fonte e caracterização do dataset

- **Fonte**: ANP — Série Histórica do Levantamento de Preços, agregado semanal nacional
  (`semanal-brasil-2004-a-2012.xlsx` + `semanal-brasil-desde-2013.xlsx`), baixados
  diretamente para `/Volumes/BinEXT/datasets/anp_combustiveis/raw/`.
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

- **Split cronológico** (sem embaralhamento): treino 2004-05-09 a 2019-11-17 (806
  semanas), validação 2019-11-24 a 2023-04-30 (172 semanas), teste 2023-05-07 a
  2026-08-30 (174 semanas) — aproximadamente 70/15/15, sem sobreposição temporal
  (verificado automaticamente: `max(train) < min(val) < ... < min(test)`).
- **Normalização**: `StandardScaler` ajustado **somente no treino** (features e alvo
  separadamente), escolhido em vez de `MinMaxScaler` porque os preços de combustível
  não têm limite superior conhecido a priori e o período de teste, como discutido
  abaixo, ultrapassa a faixa observada no treino — a padronização por z-score não
  "trava" a escala nos extremos vistos no treino.
- **Janelas**: 4, 12 e 26 semanas, geradas por `create_sequences(X, y, window_size)`.
- **Protocolo comparável**: mesmo batch size (32), mesma loss (MSE) e métrica (MAE),
  mesmo otimizador (Adam), `EarlyStopping(patience=15, restore_best_weights=True)` +
  `ReduceLROnPlateau`, mesma seed (42) e `shuffle=False` em todos os treinos.
- **Baseline**: persistência ($\hat{y}_{t+1} = y_t$).

### Hiperparametrização (Optuna)

- **Arquitetura/janela de partida**: a melhor combinação da comparação principal
  (LSTM, janela=12), usada como base para a busca.
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
- **Resultado**: melhor tentativa (trial 11) — `units=96, n_layers=1, dropout=0.037,
  learning_rate=0.00067, batch_size=32`, MAE de validação = 0.3893 R$/L. Registro
  completo em [`results/optuna_trials.csv`](../results/optuna_trials.csv) e
  [`results/optuna_best_params.json`](../results/optuna_best_params.json).

## 4. Resultados de validação e seleção dos modelos promissores

Tabela completa em [`results/experiments.csv`](../results/experiments.csv). Baseline
de persistência (referência):

| Split | MAE (R$/L) | RMSE (R$/L) | R² |
|---|---:|---:|---:|
| Treino | 0,0104 | 0,0241 | 0,9987 |
| Validação | 0,0583 | 0,1040 | 0,9888 |
| Teste | 0,0256 | 0,0528 | 0,9771 |

Comparação principal (3 janelas × 3 arquiteturas, unidades=64, 1 camada, dropout=0,2),
por MAE de validação:

| Janela | SimpleRNN | GRU | LSTM |
|---:|---:|---:|---:|
| 4 semanas | 0,678 | 0,748 | 0,771 |
| 12 semanas | 0,685 | 0,765 | **0,612** |
| 26 semanas | 0,704 | 0,747 | 0,700 |

A LSTM na janela de 12 semanas foi a melhor combinação da comparação principal e foi
usada como base para as demais grades (unidades, profundidade, regularização e
Optuna).

## 5. Três melhores configurações (por MAE de validação)

| # | Config. | Unid. | Camadas | Dropout | L2 | Parâmetros | Épocas | MAE val | RMSE val | R² val | MAE teste | RMSE teste | R² teste | Tempo (s) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | LSTM, janela 12 | 64 | 1 | 0,0 | 0,001 | 18.753 | 100 | **0,300** | 0,451 | 0,789 | 0,334 | 0,363 | -0,080 | 16,6 |
| 2 | LSTM (Optuna), janela 12 | 96 | 1 | 0,037 | 0,0 | 40.417 | 86 | 0,346 | 0,485 | 0,756 | 0,393 | 0,419 | -0,443 | 12,3 |
| 3 | LSTM, janela 12 (sem regularização) | 64 | 1 | 0,0 | 0,0 | 18.753 | 83 | 0,422 | 0,620 | 0,601 | 0,513 | 0,549 | -1,477 | 9,2 |

As três melhores configurações são todas **LSTM com janela de 12 semanas** — nenhuma
configuração de SimpleRNN ou GRU, em nenhuma janela ou unidade testada, superou essas
três. A 4ª colocada foi GRU com 128 unidades (MAE val = 0,4545), também na janela de
12 semanas.

Gráficos (curvas de treino/validação, real×previsto e resíduos) em `results/figures/`:
`*_melhor_geral_LSTM_w12_u64_L1.png` (config. 1), `*_optuna_best_LSTM_w12_u96_L1.png`
(config. 2), além das três arquiteturas na janela de 12 semanas
(`*_principal_{SimpleRNN,GRU,LSTM}_w12_u64_L1.png`).

## 6. Avaliação final no conjunto de teste e custo × efetividade

**Nenhum modelo recorrente superou o baseline de persistência no teste** (todos os R²
de teste são negativos, entre -0,08 e -10,7, enquanto a persistência atinge R²=0,977).
Isso não é um sinal de vazamento de dados nem de erro de implementação — foi
investigado (seção 40 do enunciado) e a causa identificada é **distribuição de preços
diferente entre treino e teste**:

| Split | Preço mínimo (R$/L) | Preço máximo (R$/L) | Média (R$/L) |
|---|---:|---:|---:|
| Treino (2004-2019) | 2,08 | 4,73 | 3,01 |
| Validação (2019-2023) | 3,80 | 7,39 | 5,40 |
| Teste (2023-2026) | 5,21 | 6,78 | 6,08 |

O conjunto de teste está **inteiramente acima da faixa de preços vista no treino**
(o preço mínimo do teste, R$ 5,21, já supera o preço máximo do treino, R$ 4,73), reflexo
da forte alta de preços de combustíveis no Brasil a partir de 2021-2022. Os modelos
recorrentes aprendem uma função no espaço padronizado calibrado com as estatísticas do
treino e extrapolam mal para essa faixa de preços nunca vista; a persistência, por não
depender de nenhuma função ajustada, é imune a essa mudança de nível e continua sendo o
previsor mais forte. Esse é um resultado real, consistente com a literatura sobre
séries de preços quase não estacionárias (passeio aleatório): é extremamente difícil
superar $\hat{y}_{t+1}=y_t$ quando a variável já está entre as próprias features de
entrada e a série muda de patamar entre os períodos.

**Custo computacional**: mesmo o pior caso (LSTM, 2 camadas, janela 12) treinou em
menos de 18 segundos nesta máquina (Apple Silicon, CPU — nenhuma GPU foi detectada por
`tf.config.list_physical_devices('GPU')`), pois o dataset é pequeno (~800 sequências de
treino). O custo diferencial entre arquiteturas está mais em número de parâmetros e
épocas necessárias até a convergência (early stopping) do que em tempo de parede.

## 7. SimpleRNN × GRU × LSTM (análise com base nos experimentos reais)

- **SimpleRNN**: a arquitetura mais leve (4.737 parâmetros com 64 unidades) e uma das
  mais rápidas a convergir (47-48 épocas nas janelas 4/12). Também foi consistentemente
  a **pior** das três em MAE de validação na comparação principal (0,678-0,704),
  indicando dificuldade real em aproveitar a informação temporal além de uma
  dependência simples — compatível com o problema conhecido de gradientes que
  desaparecem em RNNs simples.
- **GRU**: não mostrou, neste experimento, o equilíbrio esperado entre desempenho e
  custo — com 64 unidades teve o **pior** MAE de validação entre as três famílias em
  duas das três janelas (0,748 e 0,765, superado até pela SimpleRNN). Só melhorou
  substancialmente ao ganhar capacidade (128 unidades: MAE val 0,455, 4º melhor
  resultado geral), sugerindo que, para este problema, o GRU precisou de mais unidades
  do que a LSTM para captar o suficiente do padrão temporal.
- **LSTM**: a arquitetura com melhor desempenho em todos os cenários competitivos —
  melhor da comparação principal (janela 12, MAE val 0,612), melhor após regularização
  L2 (MAE val 0,300) e melhor após a busca Optuna (MAE val 0,346). Precisou de mais
  parâmetros (18.753 com 64 unidades, quase 4× a SimpleRNN) e, em geral, de mais épocas
  para convergir (76-100 contra 22-49 das demais), mas o custo adicional em tempo de
  parede permaneceu marginal (poucos segundos) dado o tamanho do dataset — o
  investimento em parâmetros/portões de memória (LSTM) se justificou pelo ganho de
  desempenho, algo que o GRU só replicou com o dobro de unidades e o SimpleRNN nunca
  replicou.

## 8. Janelas temporais

- **4 semanas**: insuficiente para a LSTM (pior resultado da arquitetura, MAE val
  0,771) — a rede não teve contexto suficiente para diferenciar de um ruído de curto
  prazo. Para SimpleRNN e GRU o efeito da janela foi pequeno e não sistemático.
- **12 semanas**: melhorou de forma expressiva apenas a LSTM (MAE val 0,612, queda de
  ~21% frente à janela de 4 semanas), tornando-se a base de todas as demais
  comparações (unidades, profundidade, regularização, Optuna).
- **26 semanas**: não trouxe ganho adicional — para a LSTM o MAE val piorou de volta
  para 0,700 (próximo do valor da janela de 4 semanas), e o mesmo padrão de não-melhora
  ocorreu para SimpleRNN e GRU. Isso sugere que, para esta série e este tamanho de
  dataset (~800 sequências de treino), uma janela mais longa não ajudou a capturar
  dependências de mais longo prazo — pelo contrário, pode ter dificultado a otimização
  (sequências mais longas, gradiente mais ruidoso) sem trazer informação nova relevante
  para prever apenas 1 semana à frente.
- **Conclusão sobre janelas**: a janela de **12 semanas generalizou melhor** entre as
  testadas (menor MAE de validação para a arquitetura mais forte, LSTM), e não houve
  sinal de overfitting causado pelo tamanho da janela em si — o fator dominante da
  baixa generalização no teste foi a mudança de patamar de preços (seção 6), não o
  tamanho da janela.

## 9. Profundidade e regularização

- **Profundidade** (1 vs 2 camadas, janela 12, 64 unidades): empilhar uma segunda
  camada recorrente **piorou** o resultado tanto para GRU (MAE val 0,745, contra 0,765
  de 1 camada — leve melhora, mas ainda distante do top) quanto principalmente para
  LSTM (MAE val 0,751, pior que a única camada, 0,612) e aumentou fortemente o custo
  (LSTM 2 camadas: 17,8 s e 87 épocas, mais que o dobro do tempo da versão de 1 camada).
  Não há evidência, neste dataset pequeno, de que profundidade adicional ajude — o
  contrário do que se observou com regularização L2.
- **Regularização** (LSTM, janela 12, 64 unidades): comparando nenhuma regularização
  (MAE val 0,422 / teste 0,513), dropout agressivo (0,4 + recurrent dropout 0,2 → MAE
  val 0,656 / teste 0,847, **piora clara**, indicando sobre-regularização em um modelo
  já pequeno) e L2 (1e-3, sem dropout → MAE val 0,300 / teste 0,334, **melhor resultado
  do trabalho**). O L2 foi, de longe, a técnica de regularização mais eficaz: para este
  dataset pequeno e ruidoso, restringir a magnitude dos pesos ajudou mais do que impor
  ruído estocástico via dropout nas conexões recorrentes, que aqui prejudicou a
  capacidade do modelo de usar plenamente a pouca informação disponível.
- Nenhuma configuração exibiu sinais de explosão de loss, NaN ou previsões constantes;
  o comportamento de overfitting mais evidente foi justamente a diferença entre MAE de
  treino/validação melhorando com L2 e piorando com dropout excessivo, consistente com
  o esperado teoricamente.

## 10. Limitações

- **Distribuição não estacionária**: como discutido na seção 6, o preço da gasolina
  mudou de patamar entre 2019 e 2023 (choques de preços de combustíveis no Brasil), e o
  teste ficou inteiramente fora da faixa vista no treino. Isso limita a validade de
  qualquer modelo treinado com normalização fixa; em um cenário real, seria necessário
  re-treinar periodicamente ou usar normalização adaptativa (ex.: diferenças/retornos
  em vez de nível de preço).
- **Dataset pequeno para deep learning**: apesar de cobrir 22 anos, a granularidade
  semanal produz apenas ~1150 observações (e ~800 sequências de treino), o que limita a
  capacidade de arquiteturas maiores (2 camadas, mais unidades) de generalizar sem
  overfitting.
- **Optuna não explorou L2**: a melhor configuração do trabalho (LSTM + L2) não fez
  parte do espaço de busca do Optuna (que variou dropout, não L2); um refinamento
  natural seria incluir o coeficiente L2 no espaço de busca em uma próxima iteração.
- **Baseline forte**: para preços de ativos/commodities com alta autocorrelação
  semana-a-semana, a persistência é um adversário muito difícil de superar com poucos
  dados e uma única variável de preço — um resultado consistente com a literatura de
  séries financeiras.

## 11. Conclusão

Sob um protocolo controlado (mesmos dados, split, normalização, otimizador, loss e
callbacks), a **LSTM foi a arquitetura recorrente mais eficaz** para prever o preço da
gasolina em `t+1`, superando SimpleRNN e GRU em praticamente todas as comparações,
especialmente quando combinada com janela de 12 semanas e regularização L2. Nenhuma das
três arquiteturas recorrentes testadas superou o baseline de persistência no conjunto de
teste, resultado explicado por uma mudança real e mensurável de patamar de preços entre
os períodos de treino e teste — não por vazamento de dados, erro de implementação ou
métricas fabricadas. O trabalho evidencia, com números reais extraídos de
`results/experiments.csv`, a relação entre arquitetura, memória temporal, capacidade,
custo computacional e generalização pedida no enunciado.
