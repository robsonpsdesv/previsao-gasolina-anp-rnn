"""Executa a grade de experimentos obrigatória e grava results/experiments.csv.

Grupos de experimentos (todos reais, executados nesta máquina):
  1. baseline de persistência (train/val/test)
  2. comparação principal: 3 janelas x 3 arquiteturas (SimpleRNN, GRU, LSTM)
  3. comparação de unidades (32/64/128) na janela de 12 semanas
  4. comparação de profundidade (1 vs 2 camadas empilhadas) na janela de 12 semanas
  5. comparação de regularização (nenhuma / dropout / L2) no melhor candidato
  6. busca de hiperparâmetros com Optuna

Uso: .venv/bin/python3 -m src.run_experiments
"""
import json
import time

import numpy as np
import pandas as pd
import optuna

from . import config
from .evaluation import (
    plot_prediction_vs_real,
    plot_residuals,
    plot_training_curves,
    regression_metrics,
)
from .training import persistence_baseline, train_experiment
from .utils import environment_report, set_seeds

rows = []


def _add_row(tag, cell_type, window_size, units, n_layers, dropout, l2, result, val_metrics, test_metrics):
    rows.append({
        "tag": tag,
        "modelo": cell_type,
        "janela": window_size,
        "unidades": units,
        "camadas": n_layers,
        "dropout": dropout,
        "l2": l2,
        "parametros": result["n_params"],
        "epocas": result["n_epochs_run"],
        "mae_val": val_metrics["mae"],
        "rmse_val": val_metrics["rmse"],
        "r2_val": val_metrics["r2"],
        "mae_test": test_metrics["mae"],
        "rmse_test": test_metrics["rmse"],
        "r2_test": test_metrics["r2"],
        "tempo_treino_s": result["train_time_seconds"],
        "tempo_por_epoca_s": result["train_time_per_epoch"],
    })


def run_and_record(tag, cell_type, window_size, units=64, n_layers=1, dropout=0.0,
                    recurrent_dropout=0.0, l2=0.0, save_plots=False, **kwargs):
    print(f"\n=== [{tag}] {cell_type} janela={window_size} unidades={units} camadas={n_layers} "
          f"dropout={dropout} l2={l2} ===")
    result, art = train_experiment(
        cell_type=cell_type,
        window_size=window_size,
        units=units,
        n_layers=n_layers,
        dropout=dropout,
        recurrent_dropout=recurrent_dropout,
        l2=l2,
        **kwargs,
    )
    val_metrics = regression_metrics(art["y_val"], art["y_val_pred"], art["target_scaler"])
    test_metrics = regression_metrics(art["y_test"], art["y_test_pred"], art["target_scaler"])
    print(f"  -> params={result['n_params']} epocas={result['n_epochs_run']} "
          f"tempo={result['train_time_seconds']:.1f}s | val MAE={val_metrics['mae']:.4f} "
          f"RMSE={val_metrics['rmse']:.4f} R2={val_metrics['r2']:.4f} | "
          f"test MAE={test_metrics['mae']:.4f} RMSE={test_metrics['rmse']:.4f} R2={test_metrics['r2']:.4f}")

    _add_row(tag, cell_type, window_size, units, n_layers, dropout, l2, result, val_metrics, test_metrics)

    if save_plots:
        name = f"{tag}_{cell_type}_w{window_size}_u{units}_L{n_layers}"
        plot_training_curves(art["history"], name, config.FIGURES_DIR / f"curvas_{name}.png")
        y_test_true = art["target_scaler"].inverse_transform(art["y_test"])
        y_test_pred = art["target_scaler"].inverse_transform(art["y_test_pred"])
        plot_prediction_vs_real(
            art["dates_test"], y_test_true, y_test_pred,
            f"{name} - Teste: real x previsto", config.FIGURES_DIR / f"real_vs_previsto_{name}.png",
        )
        plot_residuals(
            art["dates_test"], y_test_true.ravel(), y_test_pred.ravel(),
            name, config.FIGURES_DIR / f"residuos_{name}.png",
        )

    return result, art, val_metrics, test_metrics


def run_baseline():
    print("\n=== [baseline] Persistência (y_hat_t+1 = y_t) ===")
    base = persistence_baseline()
    for split_name in ("train", "val", "test"):
        y_true, y_pred = base[split_name]
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        mae = mean_absolute_error(y_true, y_pred)
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        r2 = r2_score(y_true, y_pred)
        print(f"  {split_name}: MAE={mae:.4f} RMSE={rmse:.4f} R2={r2:.4f}")
        if split_name in ("val", "test"):
            rows.append({
                "tag": "baseline",
                "modelo": "Persistencia",
                "janela": 1,
                "unidades": None,
                "camadas": None,
                "dropout": None,
                "l2": None,
                "parametros": 0,
                "epocas": None,
                f"mae_{split_name}": mae,
                f"rmse_{split_name}": rmse,
                f"r2_{split_name}": r2,
                "tempo_treino_s": 0.0,
                "tempo_por_epoca_s": 0.0,
            })


def main():
    set_seeds()
    print("=== Ambiente ===")
    for k, v in environment_report().items():
        print(f"{k}: {v}")

    t0 = time.perf_counter()

    # 1) Baseline
    run_baseline()

    # 2) Comparação principal: 3 janelas x 3 arquiteturas (unidades=64, 1 camada, dropout=0.2)
    best_val_mae = np.inf
    best_cfg = None
    for window in config.WINDOW_SIZES:
        for cell_type in ("SimpleRNN", "GRU", "LSTM"):
            result, art, val_m, test_m = run_and_record(
                "principal", cell_type, window, units=64, n_layers=1, dropout=0.2,
                save_plots=(window == 12),
            )
            if val_m["mae"] < best_val_mae:
                best_val_mae = val_m["mae"]
                best_cfg = dict(cell_type=cell_type, window=window)

    print(f"\n[run_experiments] Melhor config da comparação principal (por MAE de validação): {best_cfg}")

    # 3) Comparação de unidades na janela de 12 semanas
    for cell_type in ("SimpleRNN", "GRU", "LSTM"):
        for units in config.UNIT_OPTIONS:
            if units == 64:
                continue  # já coberto na comparação principal
            run_and_record("unidades", cell_type, 12, units=units, n_layers=1, dropout=0.2)

    # 4) Comparação de profundidade (1 vs 2 camadas) na janela de 12 semanas
    for cell_type in ("GRU", "LSTM"):
        run_and_record("profundidade", cell_type, 12, units=64, n_layers=2, dropout=0.2)

    # 5) Comparação de regularização no melhor candidato da comparação principal
    best_cell, best_window = best_cfg["cell_type"], best_cfg["window"]
    run_and_record("regularizacao", best_cell, best_window, units=64, n_layers=1,
                    dropout=0.0, l2=0.0)
    run_and_record("regularizacao", best_cell, best_window, units=64, n_layers=1,
                    dropout=0.4, recurrent_dropout=0.2)
    run_and_record("regularizacao", best_cell, best_window, units=64, n_layers=1,
                    dropout=0.0, l2=1e-3)

    # 6) Busca de hiperparâmetros com Optuna para a melhor arquitetura/janela
    print(f"\n=== [optuna] Busca de hiperparâmetros para {best_cell}, janela={best_window} ===")

    def objective(trial):
        units = trial.suggest_categorical("units", [16, 32, 64, 96, 128])
        n_layers = trial.suggest_int("n_layers", 1, 2)
        dropout = trial.suggest_float("dropout", 0.0, 0.5)
        learning_rate = trial.suggest_float("learning_rate", 1e-4, 5e-3, log=True)
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])

        result, art = train_experiment(
            cell_type=best_cell,
            window_size=best_window,
            units=units,
            n_layers=n_layers,
            dropout=dropout,
            learning_rate=learning_rate,
            batch_size=batch_size,
            max_epochs=60,
            patience=10,
            seed=config.SEED,
        )
        val_metrics = regression_metrics(art["y_val"], art["y_val_pred"], art["target_scaler"])
        trial.set_user_attr("n_params", result["n_params"])
        trial.set_user_attr("val_rmse", val_metrics["rmse"])
        trial.set_user_attr("val_r2", val_metrics["r2"])
        return val_metrics["mae"]

    sampler = optuna.samplers.TPESampler(seed=config.SEED)
    study = optuna.create_study(direction="minimize", sampler=sampler, study_name="anp_rnn_search")
    study.optimize(objective, n_trials=20, show_progress_bar=False)

    print(f"[optuna] Melhor trial: {study.best_trial.number} valor(MAE val)={study.best_value:.4f}")
    print(f"[optuna] Melhores hiperparâmetros: {study.best_params}")

    optuna_trials_df = study.trials_dataframe()
    optuna_trials_df.to_csv(config.RESULTS_DIR / "optuna_trials.csv", index=False)

    with open(config.RESULTS_DIR / "optuna_best_params.json", "w") as f:
        json.dump({
            "best_value_mae_val": study.best_value,
            "best_params": study.best_params,
            "best_trial_number": study.best_trial.number,
            "n_trials": len(study.trials),
            "cell_type": best_cell,
            "window_size": best_window,
        }, f, indent=2, ensure_ascii=False)

    # Treina o modelo final com os melhores hiperparâmetros do Optuna e registra na tabela
    best_params = study.best_params
    run_and_record(
        "optuna_best", best_cell, best_window,
        units=best_params["units"], n_layers=best_params["n_layers"],
        dropout=best_params["dropout"], learning_rate=best_params["learning_rate"],
        batch_size=best_params["batch_size"], save_plots=True,
    )

    # --- consolida experiments.csv ---
    df_results = pd.DataFrame(rows)
    df_results.to_csv(config.EXPERIMENTS_CSV, index=False)
    print(f"\n[run_experiments] Tabela de experimentos salva em {config.EXPERIMENTS_CSV}")
    print(f"[run_experiments] Tempo total: {(time.perf_counter() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
