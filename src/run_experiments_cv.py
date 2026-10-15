"""Validação cruzada temporal (TimeSeriesSplit) e avaliação walk-forward com re-treino.

Parte A: CV com janela expansível no período de desenvolvimento (treino + validação do split
60/20/20), sem usar o teste. Compara arquiteturas e janelas pela média e pelo desvio do MAE em
5 folds e escolhe uma configuração por alvo.
Parte B: walk-forward no período de teste. A cada bloco de 58 semanas o modelo é re-treinado com
todo o histórico anterior; os 4 blocos cobrem as mesmas 232 semanas de teste do split 60/20/20.

Em cada fold os scalers são ajustados só no trecho de treino, o early stopping usa as últimas 26
semanas desse trecho (assim o ajuste inclui o patamar de preços mais recente) e o bloco avaliado
nunca entra no treino. Grava
results/cv_temporal.csv, results/walk_forward.csv e results/figures/walk_forward_real_vs_previsto.png.

Uso: .venv/bin/python3 -m src.run_experiments_cv
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from . import config
from .evaluation import inverse_target, price_metrics
from .models import ModelConfig
from .preprocessing import FEATURE_COLUMNS, TARGET_COLUMN, chronological_split
from .sequences import create_sequences
from .training import fit_model, load_or_build_processed_df, to_log_return_frame, train_experiment
from .utils import set_seeds

CELLS = ("SimpleRNN", "GRU", "LSTM")
MODES = ("level", "delta")
CV_SPLITS, CV_BLOCK = 5, 150  # 5 blocos de 150 semanas no fim do período de desenvolvimento
WF_SPLITS, WF_BLOCK = 4, 58  # 4 x 58 = 232 semanas, o teste do split 60/20/20
INNER_VAL_WEEKS = 26
MODE_LABELS = {"level": "Nível de preço", "delta": "Log-retorno"}


def load_sequences(mode: str, window: int):
    """Janelas sem escala, preço em t, preço real em t+1 e data-base t de cada amostra."""
    df = load_or_build_processed_df()
    if mode == "delta":
        df = to_log_return_frame(df)
        price_t, price_next = df["preco_t"].values, df["preco_t_mais_1"].values
    else:
        price_t, price_next = df["gasolina_preco_medio"].values, df[TARGET_COLUMN].values
    X, y = create_sequences(df[FEATURE_COLUMNS].values, df[[TARGET_COLUMN]].values, window)
    keep = slice(window - 1, None)
    return X, y, price_t[keep], price_next[keep], df.index[keep]


def fit_and_predict(cell, mode, X, y, price_t, train_idx, eval_idx):
    """Treina no histórico do fold e prevê o bloco seguinte em R$/L."""
    set_seeds(config.SEED)
    fit_idx, es_idx = train_idx[:-INNER_VAL_WEEKS], train_idx[-INNER_VAL_WEEKS:]
    # janelas consecutivas avançam uma linha: estas são as linhas da série vistas no ajuste
    rows = np.concatenate([X[fit_idx[0]], X[fit_idx[1:], -1]])
    fs, ts = StandardScaler().fit(rows), StandardScaler().fit(y[fit_idx])

    def scaled(idx):
        return ((X[idx] - fs.mean_) / fs.scale_).astype(np.float32), ts.transform(y[idx]).astype(np.float32)

    cfg = ModelConfig(cell_type=cell, units=64, n_layers=1, dropout=0.2, learning_rate=config.LEARNING_RATE)
    model, history, elapsed, epoch_time = fit_model(cfg, *scaled(fit_idx), *scaled(es_idx))
    pred = ts.inverse_transform(model.predict(scaled(eval_idx)[0], verbose=0)).ravel()
    if mode == "delta":
        pred = price_t[eval_idx] * np.exp(pred)
    info = {"epocas": len(history.history["loss"]), "tempo_treino_s": elapsed,
            "tempo_epoca_mediano_s": epoch_time}
    return pred, info


def cross_validation(dev_end) -> pd.DataFrame:
    """Parte A: TimeSeriesSplit restrito às amostras com data-base até o fim da validação."""
    rows = []
    for mode in MODES:
        for window in config.WINDOW_SIZES:
            X, y, price_t, price_next, dates = load_sequences(mode, window)
            n_dev = int((dates <= dev_end).sum())
            splitter = TimeSeriesSplit(n_splits=CV_SPLITS, test_size=CV_BLOCK)
            for fold, (train_idx, eval_idx) in enumerate(splitter.split(X[:n_dev]), start=1):
                base = price_metrics(price_next[eval_idx], price_t[eval_idx])
                for cell in CELLS:
                    pred, info = fit_and_predict(cell, mode, X, y, price_t, train_idx, eval_idx)
                    m = price_metrics(price_next[eval_idx], pred)
                    print(f"[cv/{mode}] {cell} w{window} fold {fold}: MAE={m['mae']:.4f} "
                          f"(persistência {base['mae']:.4f}) épocas={info['epocas']}")
                    rows.append({
                        "alvo": mode, "modelo": cell, "janela": window, "fold": fold,
                        "treino_ate": dates[train_idx[-1]].date(),
                        "bloco_inicio": dates[eval_idx[0]].date(), "bloco_fim": dates[eval_idx[-1]].date(),
                        "n_treino": len(train_idx),
                        "preco_max_treino": price_t[train_idx].max(), "preco_max_bloco": price_next[eval_idx].max(),
                        "mae": m["mae"], "rmse": m["rmse"],
                        "mae_persistencia": base["mae"], "rmse_persistencia": base["rmse"], **info,
                    })
    return pd.DataFrame(rows)


def walk_forward():
    """Parte B: re-treino a cada bloco de 58 semanas sobre o período de teste do split 60/20/20."""
    rows, series = [], {}
    for mode in MODES:
        for window in config.WINDOW_SIZES:
            X, y, price_t, price_next, dates = load_sequences(mode, window)
            folds = list(TimeSeriesSplit(n_splits=WF_SPLITS, test_size=WF_BLOCK).split(X))
            test_idx = np.concatenate([e for _, e in folds])
            base = price_metrics(price_next[test_idx], price_t[test_idx])
            base_blocks = {f"mae_persistencia_bloco{k}": price_metrics(price_next[e], price_t[e])["mae"]
                           for k, (_, e) in enumerate(folds, start=1)}
            for cell in CELLS:
                preds, infos = [], []
                for train_idx, eval_idx in folds:
                    pred, info = fit_and_predict(cell, mode, X, y, price_t, train_idx, eval_idx)
                    preds.append(pred)
                    infos.append(info)
                pred = np.concatenate(preds)
                m = price_metrics(price_next[test_idx], pred)
                print(f"[walk-forward/{mode}] {cell} w{window}: MAE={m['mae']:.4f} "
                      f"(persistência {base['mae']:.4f})")
                rows.append({
                    "alvo": mode, "modelo": cell, "janela": window,
                    "mae_test": m["mae"], "rmse_test": m["rmse"], "r2_test": m["r2"],
                    "mae_persistencia": base["mae"], "rmse_persistencia": base["rmse"],
                    "r2_persistencia": base["r2"],
                    **{f"mae_bloco{k}": price_metrics(price_next[e], p)["mae"]
                       for k, ((_, e), p) in enumerate(zip(folds, preds), start=1)},
                    **base_blocks,
                    "epocas_media": float(np.mean([i["epocas"] for i in infos])),
                    "tempo_treino_s": sum(i["tempo_treino_s"] for i in infos),
                    "tempo_epoca_mediano_s": float(np.median([i["tempo_epoca_mediano_s"] for i in infos])),
                })
                series[(mode, cell, window)] = (dates[test_idx], price_next[test_idx], pred,
                                                [dates[e[0]] for _, e in folds])
    return pd.DataFrame(rows), series


def plot_walk_forward(series, chosen, holdout, out_path):
    fig, axes = plt.subplots(len(chosen), 1, figsize=(11, 3.8 * len(chosen)), sharex=True)
    for ax, key in zip(np.atleast_1d(axes), chosen):
        mode, cell, window = key
        dates, real, pred, starts = series[key]
        ax.plot(dates, real, label="real", linewidth=1.3)
        ax.plot(dates, pred, "--", label="walk-forward (re-treino a cada 58 semanas)", linewidth=1.2)
        ax.plot(dates, holdout[key], ":", label="hold-out 60/20/20 (treino único até 2017-08)", linewidth=1.4)
        for d in starts:
            ax.axvline(d, color="gray", linewidth=0.7, linestyle=":")
        ax.set_title(f"{MODE_LABELS[mode]}: {cell}, janela {window} (escolhida pela CV temporal)")
        ax.set_ylabel("R$/litro")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main():
    set_seeds()
    dev_end = chronological_split(load_or_build_processed_df())[1].index.max()

    cv = cross_validation(dev_end)
    cv.to_csv(config.RESULTS_DIR / "cv_temporal.csv", index=False)
    summary = (cv.groupby(["alvo", "modelo", "janela"], sort=False)
                 .agg(mae_cv_media=("mae", "mean"), mae_cv_desvio=("mae", "std"),
                      mae_cv_persistencia=("mae_persistencia", "mean"))
                 .reset_index())
    print(summary.round(4).to_string(index=False))
    # escolha por alvo feita só com a CV (o teste não participa)
    chosen = summary.loc[summary.groupby("alvo", sort=False)["mae_cv_media"].idxmin()]
    chosen_keys = [(r.alvo, r.modelo, int(r.janela)) for r in chosen.itertuples()]

    wf, series = walk_forward()
    wf = wf.merge(summary, on=["alvo", "modelo", "janela"])
    wf["escolhida_pela_cv"] = [k in chosen_keys for k in zip(wf["alvo"], wf["modelo"], wf["janela"])]
    wf.to_csv(config.RESULTS_DIR / "walk_forward.csv", index=False)
    print(wf[["alvo", "modelo", "janela", "mae_cv_media", "mae_test", "mae_persistencia",
              "rmse_test", "rmse_persistencia", "escolhida_pela_cv"]].round(4).to_string(index=False))

    holdout = {}
    for key in chosen_keys:
        mode, cell, window = key
        _, art = train_experiment(cell, window, units=64, n_layers=1, dropout=0.2, target_mode=mode)
        holdout[key] = (art["price_pred_test"] if mode == "delta"
                        else inverse_target(art["y_test_pred"], art["target_scaler"]))
    plot_walk_forward(series, chosen_keys, holdout, config.FIGURES_DIR / "walk_forward_real_vs_previsto.png")
    print(f"[cv] resultados em {config.RESULTS_DIR / 'cv_temporal.csv'} e {config.RESULTS_DIR / 'walk_forward.csv'}")


if __name__ == "__main__":
    main()
