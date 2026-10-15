"""Diagnóstico da saturação das ativações (tanh/sigmoid) nos modelos salvos em results/models.

Quando as entradas ficam muito fora da faixa do treino, as pré-ativações das portas e dos
candidatos crescem e as derivadas de sigmoide/tanh tendem a zero: a rede deixa de reagir às
entradas. Para cada modelo e partição são medidos a faixa do preço de entrada em unidades do scaler
do treino (z), a sensibilidade da saída às entradas (norma do gradiente dŷ/dX na escala
padronizada), a inclinação da previsão em relação ao preço em t e a dispersão das previsões contra
a dos valores reais. Gera results/diagnostico_saturacao.csv e results/figures/diagnostico_saturacao.png.

Uso: .venv/bin/python3 -m src.diagnostics  (após run_experiments e run_experiments_delta)
"""
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

from . import config
from .preprocessing import FEATURE_COLUMNS, TARGET_COLUMN
from .training import load_or_build_processed_df, prepare_window_data, to_log_return_frame

PRICE_COL = FEATURE_COLUMNS.index("gasolina_preco_medio")
SPLIT_LABELS = {"train": "treino", "val": "validação", "test": "teste"}
SPLIT_COLORS = {"train": "tab:blue", "val": "tab:orange", "test": "tab:green"}


def analyse(path):
    mode = "delta" if path.stem.startswith("delta_") else "level"
    window = int(re.search(r"_w(\d+)_", path.stem).group(1))
    model = keras.models.load_model(path)
    splits, _, target_scaler = prepare_window_data(window, target_mode=mode)[:3]
    df = load_or_build_processed_df()
    frame = (to_log_return_frame(df) if mode == "delta"
             else df.assign(preco_t=df["gasolina_preco_medio"], preco_t_mais_1=df[TARGET_COLUMN]))

    rows, points = [], {}
    for split, (X, _, dates) in splits.items():
        x = tf.convert_to_tensor(X)
        with tf.GradientTape() as tape:
            tape.watch(x)
            out_z = model(x, training=False)
        # cada saída depende só da própria janela, então o gradiente da soma é o gradiente por amostra
        sensitivity = np.linalg.norm(tape.gradient(out_z, x).numpy().reshape(len(X), -1), axis=1)
        out = target_scaler.inverse_transform(np.asarray(out_z)).ravel()
        price_t = frame.loc[dates, "preco_t"].values
        real = frame.loc[dates, "preco_t_mais_1"].values
        pred = price_t * np.exp(out) if mode == "delta" else out
        rows.append({
            "modelo": path.stem, "alvo": mode, "particao": SPLIT_LABELS[split],
            "z_preco_entrada_min": X[:, -1, PRICE_COL].min(), "z_preco_entrada_max": X[:, -1, PRICE_COL].max(),
            "sensibilidade_mediana": float(np.median(sensitivity)),
            "inclinacao_previsto_vs_preco_t": float(np.polyfit(price_t, pred, 1)[0]),
            "real_min": real.min(), "real_max": real.max(), "real_desvio": real.std(),
            "previsto_min": pred.min(), "previsto_max": pred.max(), "previsto_desvio": pred.std(),
            "mae": float(np.abs(real - pred).mean()),
        })
        points[split] = (price_t, pred, sensitivity)
    return rows, points


def plot_saturation(points_by_model, out_path):
    fig, axes = plt.subplots(len(points_by_model), 2, figsize=(12, 4.3 * len(points_by_model)))
    for row, (name, points) in zip(np.atleast_2d(axes), points_by_model.items()):
        lo = min(p[0].min() for p in points.values())
        hi = max(p[0].max() for p in points.values())
        for split, (price_t, pred, sensitivity) in points.items():
            kw = dict(s=6, alpha=0.6, color=SPLIT_COLORS[split], label=SPLIT_LABELS[split])
            row[0].scatter(price_t, pred, **kw)
            row[1].scatter(price_t, sensitivity, **kw)
        row[0].plot([lo, hi], [lo, hi], color="black", linewidth=0.8, label="persistência (previsão = preço em t)")
        row[0].set_xlabel("preço em t (R$/L)")
        row[0].set_ylabel("previsão para t+1 (R$/L)")
        row[0].set_title(f"{name}\nprevisão × preço de entrada", fontsize=10)
        row[0].legend(fontsize=8)
        row[1].set_yscale("log")
        row[1].set_xlabel("preço em t (R$/L)")
        row[1].set_ylabel("‖dŷ/dX‖ (escala padronizada)")
        row[1].set_title(f"{name}\nsensibilidade da saída às entradas", fontsize=10)
        row[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main():
    paths = sorted(config.MODELS_DIR.glob("top*_*.keras")) + sorted(config.MODELS_DIR.glob("delta_best_*.keras"))
    rows, points = [], {}
    for path in paths:
        model_rows, model_points = analyse(path)
        rows += model_rows
        if path.stem.startswith(("top1_", "delta_best_")):
            points[path.stem] = model_points
    table = pd.DataFrame(rows)
    table.to_csv(config.RESULTS_DIR / "diagnostico_saturacao.csv", index=False)
    print(table.round(3).to_string(index=False))
    plot_saturation(points, config.FIGURES_DIR / "diagnostico_saturacao.png")
    print(f"[diagnostics] {config.RESULTS_DIR / 'diagnostico_saturacao.csv'}")


if __name__ == "__main__":
    main()
