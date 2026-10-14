"""Sensibilidade ao split cronológico (treino/validação/teste) em nível e em log-retorno.

Para cada split roda 3 arquiteturas x 3 janelas (64 unidades, 1 camada, dropout 0,2) nos dois
alvos ("level" e "delta") e compara com a persistência. Grava results/experiments_splits.csv.

Uso: .venv/bin/python3 -m src.run_experiments_splits
"""
import pandas as pd

from . import config
from .evaluation import inverse_target, price_metrics
from .preprocessing import TARGET_COLUMN, chronological_split
from .training import load_or_build_processed_df, prepare_window_data, train_experiment, to_log_return_frame
from .utils import set_seeds

SPLITS = [(0.60, 0.20), (0.70, 0.15), (0.75, 0.15), (0.80, 0.10)]
MODES = ("level", "delta")

rows = []


def split_name(fracs):
    train, val = fracs
    return f"{round(train * 100)}/{round(val * 100)}/{round((1 - train - val) * 100)}"


def persistence(mode, fracs):
    df = load_or_build_processed_df()
    if mode == "delta":
        df = to_log_return_frame(df).drop(columns=TARGET_COLUMN).rename(columns={"preco_t_mais_1": TARGET_COLUMN})
        df["gasolina_preco_medio"] = df["preco_t"]
    _, val_df, test_df = chronological_split(df, *fracs)
    return {
        "val": price_metrics(val_df[TARGET_COLUMN], val_df["gasolina_preco_medio"]),
        "test": price_metrics(test_df[TARGET_COLUMN], test_df["gasolina_preco_medio"]),
    }


def main():
    set_seeds()
    for fracs in SPLITS:
        name = split_name(fracs)
        df = load_or_build_processed_df()
        train_df, val_df, test_df = chronological_split(df, *fracs)
        faixas = {
            s: (d["gasolina_preco_medio"].min(), d["gasolina_preco_medio"].max())
            for s, d in (("train", train_df), ("val", val_df), ("test", test_df))
        }
        print(f"\n##### Split {name} | treino até {train_df.index.max().date()} "
              f"| faixa de preço (min, max) treino/val/teste: {faixas}")
        ranges = {
            "treino_ate": str(train_df.index.max().date()),
            "n_treino": len(train_df), "n_val": len(val_df), "n_teste": len(test_df),
            "treino_max": faixas["train"][1], "val_min": faixas["val"][0], "teste_min": faixas["test"][0],
            "teste_max": faixas["test"][1],
        }

        for mode in MODES:
            base = persistence(mode, fracs)
            print(f"[{name}/{mode}] persistência: val MAE={base['val']['mae']:.4f} "
                  f"test MAE={base['test']['mae']:.4f} R2={base['test']['r2']:.4f}")
            rows.append({"split": name, "alvo": mode, "modelo": "Persistencia", "janela": 1,
                         "mae_val": base["val"]["mae"], "r2_val": base["val"]["r2"],
                         "mae_test": base["test"]["mae"], "rmse_test": base["test"]["rmse"],
                         "r2_test": base["test"]["r2"], **ranges})
            for window in config.WINDOW_SIZES:
                for cell in ("SimpleRNN", "GRU", "LSTM"):
                    result, art = train_experiment(
                        cell_type=cell, window_size=window, units=64, n_layers=1, dropout=0.2,
                        target_mode=mode, split_fracs=fracs,
                    )
                    if mode == "delta":
                        val = price_metrics(art["price_true_val"], art["price_pred_val"])
                        test = price_metrics(art["price_true_test"], art["price_pred_test"])
                    else:
                        ts = art["target_scaler"]
                        val = price_metrics(inverse_target(art["y_val"], ts), inverse_target(art["y_val_pred"], ts))
                        test = price_metrics(inverse_target(art["y_test"], ts), inverse_target(art["y_test_pred"], ts))
                    print(f"[{name}/{mode}] {cell} w{window}: epocas={result['n_epochs_run']} "
                          f"val MAE={val['mae']:.4f} | test MAE={test['mae']:.4f} R2={test['r2']:.4f}")
                    rows.append({"split": name, "alvo": mode, "modelo": cell, "janela": window,
                                 "epocas": result["n_epochs_run"],
                                 "mae_val": val["mae"], "r2_val": val["r2"],
                                 "mae_test": test["mae"], "rmse_test": test["rmse"],
                                 "r2_test": test["r2"], **ranges})

    out = config.RESULTS_DIR / "experiments_splits.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\n[splits] resultados em {out}")


if __name__ == "__main__":
    main()
