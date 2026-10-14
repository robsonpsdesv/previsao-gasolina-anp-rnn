"""Experimentos com alvo/features em log-retorno semanal (série estacionária).

Motivação: com o alvo em nível, validação e teste ficam fora da faixa de preços do treino
e nenhum modelo supera a persistência. Aqui a rede prevê r = log(y_{t+1}/y_t) e o preço é
reconstruído como y_t * exp(r_hat); a persistência equivale a r_hat = 0. As métricas são
sempre calculadas em R$/litro, comparáveis às de results/experiments.csv.

Uso: .venv/bin/python3 -m src.run_experiments_delta
"""
import json

import optuna
import pandas as pd

from . import config
from .evaluation import plot_prediction_vs_real, plot_training_curves, price_metrics
from .training import prepare_window_data, train_experiment
from .utils import environment_report, set_seeds

MODE = "delta"
OUT_CSV = config.RESULTS_DIR / "experiments_delta.csv"

rows = []
_runs = []


def run_and_record(tag, cell_type, window, units=64, n_layers=1, dropout=0.2, **kwargs):
    print(f"\n=== [{tag}] {cell_type} janela={window} unidades={units} camadas={n_layers} dropout={dropout} ===")
    result, art = train_experiment(
        cell_type=cell_type, window_size=window, units=units, n_layers=n_layers,
        dropout=dropout, target_mode=MODE, **kwargs,
    )
    val = price_metrics(art["price_true_val"], art["price_pred_val"])
    test = price_metrics(art["price_true_test"], art["price_pred_test"])
    print(f"  -> epocas={result['n_epochs_run']} | val MAE={val['mae']:.4f} R2={val['r2']:.4f} "
          f"| test MAE={test['mae']:.4f} RMSE={test['rmse']:.4f} R2={test['r2']:.4f}")
    rows.append({
        "tag": tag, "modelo": cell_type, "janela": window, "unidades": units, "camadas": n_layers,
        "dropout": dropout, "parametros": result["n_params"], "epocas": result["n_epochs_run"],
        "mae_val": val["mae"], "rmse_val": val["rmse"], "r2_val": val["r2"],
        "mae_test": test["mae"], "rmse_test": test["rmse"], "r2_test": test["r2"],
        "tempo_treino_s": result["train_time_seconds"],
    })
    _runs.append((val["mae"], f"{tag}_{cell_type}_w{window}_u{units}_L{n_layers}", art))
    return val, test


def persistence_rows():
    # Persistência = retorno previsto 0; usa as mesmas amostras de val/teste dos modelos.
    prices = prepare_window_data(4, target_mode=MODE)[3]
    for split in ("val", "test"):
        preco_t, preco_real = prices[split].T
        m = price_metrics(preco_real, preco_t)
        print(f"[baseline] persistência {split}: MAE={m['mae']:.4f} RMSE={m['rmse']:.4f} R2={m['r2']:.4f}")
        rows.append({"tag": "baseline", "modelo": "Persistencia", "janela": 1, "parametros": 0,
                     f"mae_{split}": m["mae"], f"rmse_{split}": m["rmse"], f"r2_{split}": m["r2"]})


def main():
    set_seeds()
    for k, v in environment_report().items():
        print(f"{k}: {v}")

    persistence_rows()

    best_val, best_cfg = float("inf"), None
    for window in config.WINDOW_SIZES:
        for cell in ("SimpleRNN", "GRU", "LSTM"):
            val, _ = run_and_record("principal", cell, window)
            if val["mae"] < best_val:
                best_val, best_cfg = val["mae"], (cell, window)
    cell, window = best_cfg
    print(f"\n[delta] Melhor da comparação principal (MAE val): {cell}, janela={window}")

    def objective(trial):
        kw = dict(
            units=trial.suggest_categorical("units", [16, 32, 64, 96, 128]),
            n_layers=trial.suggest_int("n_layers", 1, 2),
            dropout=trial.suggest_float("dropout", 0.0, 0.5),
            learning_rate=trial.suggest_float("learning_rate", 1e-4, 5e-3, log=True),
            batch_size=trial.suggest_categorical("batch_size", [16, 32, 64]),
        )
        _, art = train_experiment(cell_type=cell, window_size=window, max_epochs=60, patience=10,
                                  target_mode=MODE, **kw)
        return price_metrics(art["price_true_val"], art["price_pred_val"])["mae"]

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=config.SEED))
    study.optimize(objective, n_trials=20)
    print(f"[optuna] melhor MAE val={study.best_value:.4f} params={study.best_params}")
    study.trials_dataframe().to_csv(config.RESULTS_DIR / "optuna_delta_trials.csv", index=False)
    with open(config.RESULTS_DIR / "optuna_delta_best_params.json", "w") as f:
        json.dump({"best_value_mae_val": study.best_value, "best_params": study.best_params,
                   "cell_type": cell, "window_size": window}, f, indent=2)

    p = study.best_params
    run_and_record("optuna_best", cell, window, units=p["units"], n_layers=p["n_layers"],
                   dropout=p["dropout"], learning_rate=p["learning_rate"], batch_size=p["batch_size"])

    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
    mae_val, name, art = min(_runs, key=lambda r: r[0])
    art["model"].save(config.MODELS_DIR / f"delta_best_{name}.keras")
    plot_training_curves(art["history"], name, config.FIGURES_DIR / f"delta_curvas_{name}.png")
    plot_prediction_vs_real(art["dates_test"], art["price_true_test"], art["price_pred_test"],
                            f"{name} (retorno) - Teste: real x previsto",
                            config.FIGURES_DIR / f"delta_real_vs_previsto_{name}.png")
    print(f"[delta] melhor: {name} (MAE val={mae_val:.4f}); resultados em {OUT_CSV}")


if __name__ == "__main__":
    main()
