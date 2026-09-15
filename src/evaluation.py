"""Métricas de avaliação e gráficos (curvas de treino, real x previsto, resíduos)."""
import sys

import matplotlib

# Backend não-interativo apenas em execução via script (fora do Jupyter), para não
# quebrar a renderização inline de gráficos quando importado a partir de um notebook.
if "ipykernel" not in sys.modules:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def inverse_target(y_scaled: np.ndarray, target_scaler) -> np.ndarray:
    return target_scaler.inverse_transform(y_scaled.reshape(-1, 1)).ravel()


def regression_metrics(y_true_scaled: np.ndarray, y_pred_scaled: np.ndarray, target_scaler) -> dict:
    """Calcula MAE, RMSE e R² na escala original (R$/litro)."""
    y_true = inverse_target(y_true_scaled, target_scaler)
    y_pred = inverse_target(y_pred_scaled, target_scaler)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = r2_score(y_true, y_pred)

    return {"mae": mae, "rmse": rmse, "r2": r2}


def plot_training_curves(history, title: str, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(history["loss"], label="treino")
    axes[0].plot(history["val_loss"], label="validação")
    axes[0].set_title(f"{title} - Loss (MSE)")
    axes[0].set_xlabel("época")
    axes[0].legend()

    axes[1].plot(history["mae"], label="treino")
    axes[1].plot(history["val_mae"], label="validação")
    axes[1].set_title(f"{title} - MAE")
    axes[1].set_xlabel("época")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_prediction_vs_real(dates, y_true, y_pred, title: str, out_path):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(dates, y_true, label="real", linewidth=1.2)
    ax.plot(dates, y_pred, label="previsto", linewidth=1.2, linestyle="--")
    ax.set_title(title)
    ax.set_ylabel("R$/litro")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_residuals(dates, y_true, y_pred, title: str, out_path):
    residuals = y_true - y_pred
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(dates, residuals)
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set_title(f"{title} - Resíduos ao longo do tempo")
    axes[0].set_ylabel("erro (R$/litro)")

    axes[1].hist(residuals, bins=30)
    axes[1].set_title(f"{title} - Distribuição dos resíduos")

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
