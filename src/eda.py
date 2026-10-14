"""Análise exploratória: estatísticas, distribuições e datas das fronteiras dos splits.

Gera em results/: eda_estatisticas_features.csv, eda_estatisticas_por_split.csv,
fronteiras_splits.csv e figuras results/figures/eda_*.png.

Uso: .venv/bin/python3 -m src.eda
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config
from .preprocessing import FEATURE_COLUMNS, TARGET_COLUMN, chronological_split
from .training import load_or_build_processed_df, prepare_window_data

PRICE = "gasolina_preco_medio"
SPLIT_COLORS = {"treino": "tab:blue", "validação": "tab:orange", "teste": "tab:green"}
SPLITS = [(0.60, 0.20), (0.70, 0.15), (0.75, 0.15), (0.80, 0.10)]


def _acf(x: np.ndarray, max_lag: int) -> np.ndarray:
    x = x - x.mean()
    denom = float(np.dot(x, x))
    return np.array([np.dot(x[:len(x) - k], x[k:]) / denom for k in range(1, max_lag + 1)])


def _splits(df, fracs=(config.TRAIN_FRAC, config.VAL_FRAC)):
    train, val, test = chronological_split(df, *fracs)
    return {"treino": train, "validação": val, "teste": test}


def describe_features(df: pd.DataFrame) -> pd.DataFrame:
    stats = df[FEATURE_COLUMNS].describe().T
    stats["assimetria"] = df[FEATURE_COLUMNS].skew()
    stats["curtose"] = df[FEATURE_COLUMNS].kurt()
    return stats


def describe_by_split(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    logret = np.log(df[PRICE]).diff()
    for name, part in _splits(df).items():
        p = part[PRICE]
        r = logret.loc[part.index].dropna()
        rows.append({
            "split": name, "inicio": part.index.min().date(), "fim": part.index.max().date(),
            "semanas": len(p), "preco_medio": p.mean(), "preco_desvio": p.std(),
            "preco_min": p.min(), "preco_q25": p.quantile(.25), "preco_mediana": p.median(),
            "preco_q75": p.quantile(.75), "preco_max": p.max(),
            "retorno_medio": r.mean(), "retorno_desvio": r.std(),
            "retorno_assimetria": r.skew(), "retorno_curtose": r.kurt(),
            "retorno_acf_lag1": _acf(r.values, 1)[0],
        })
    return pd.DataFrame(rows)


def plot_series_and_splits(df, out):
    fig, ax = plt.subplots(figsize=(11, 4))
    for name, part in _splits(df).items():
        ax.plot(part.index, part[PRICE], color=SPLIT_COLORS[name], label=name)
    ax.set_ylabel("preço médio da gasolina (R$/L)")
    ax.set_title("Série semanal e split cronológico 60/20/20")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def plot_price_distributions(df, out):
    parts = _splits(df)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    bins = np.linspace(df[PRICE].min(), df[PRICE].max(), 30)
    for name, part in parts.items():
        axes[0].hist(part[PRICE], bins=bins, alpha=0.6, color=SPLIT_COLORS[name], label=name)
    axes[0].set_title("Distribuição do preço por split")
    axes[0].set_xlabel("R$/L")
    axes[0].legend()
    axes[1].boxplot([p[PRICE] for p in parts.values()], tick_labels=list(parts))
    axes[1].set_title("Boxplot do preço por split")
    axes[1].set_ylabel("R$/L")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def plot_return_distributions(df, out):
    logret = np.log(df[PRICE]).diff().dropna()
    parts = _splits(df)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    bins = np.linspace(logret.quantile(.005), logret.quantile(.995), 40)
    for name, part in parts.items():
        axes[0].hist(logret.loc[logret.index.intersection(part.index)], bins=bins, alpha=0.6,
                     color=SPLIT_COLORS[name], label=name, density=True)
    axes[0].set_title("Log-retorno semanal por split")
    axes[0].set_xlabel("log(p_t / p_{t-1})")
    axes[0].legend()
    axes[1].boxplot([logret.loc[logret.index.intersection(p.index)] for p in parts.values()],
                    tick_labels=list(parts))
    axes[1].set_title("Boxplot do log-retorno por split")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def plot_feature_histograms(df, out):
    fig, axes = plt.subplots(2, 4, figsize=(15, 6))
    for ax, col in zip(axes.ravel(), FEATURE_COLUMNS):
        ax.hist(df[col], bins=30, color="tab:gray")
        ax.set_title(col, fontsize=9)
    fig.suptitle("Distribuição das 8 features (série completa)")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def plot_acf(df, out, max_lag=26):
    price = df[PRICE].values
    logret = np.log(df[PRICE]).diff().dropna().values
    lags = np.arange(1, max_lag + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, series, title in ((axes[0], price, "preço (nível)"), (axes[1], logret, "log-retorno semanal")):
        ax.bar(lags, _acf(series, max_lag))
        band = 1.96 / np.sqrt(len(series))
        ax.axhline(band, color="red", linestyle="--", linewidth=0.8)
        ax.axhline(-band, color="red", linestyle="--", linewidth=0.8)
        ax.set_title(f"Autocorrelação: {title}")
        ax.set_xlabel("defasagem (semanas)")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def boundary_table() -> pd.DataFrame:
    """Datas (início das entradas, base t, data do alvo) das amostras de cada partição."""
    df = load_or_build_processed_df()
    dates = df.index
    next_date = {d: n for d, n in zip(dates[:-1], dates[1:])}
    next_date[dates[-1]] = dates[-1] + pd.Timedelta(days=7)  # alvo da última linha vem do fim da série bruta
    rows = []
    for fracs in SPLITS:
        label = f"{round(fracs[0] * 100)}/{round(fracs[1] * 100)}/{round((1 - sum(fracs)) * 100)}"
        for mode in ("level", "delta"):
            for window in config.WINDOW_SIZES:
                splits = prepare_window_data(window, target_mode=mode, split_fracs=fracs)[0]
                base_index = dates if mode == "level" else dates[1:]
                for part, (X, _, seq_dates) in splits.items():
                    first_base, last_base = seq_dates.min(), seq_dates.max()
                    first_input = base_index[base_index.get_loc(first_base) - window + 1]
                    rows.append({
                        "split": label, "alvo": mode, "janela": window, "particao": part,
                        "amostras": len(X),
                        "primeira_entrada": first_input.date(), "primeira_data_base_t": first_base.date(),
                        "primeiro_alvo_t_mais_1": next_date[first_base].date(),
                        "ultima_entrada_t": last_base.date(),
                        "ultimo_alvo_t_mais_1": next_date[last_base].date(),
                    })
    return pd.DataFrame(rows)


def main():
    df = load_or_build_processed_df()
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    describe_features(df).round(4).to_csv(config.RESULTS_DIR / "eda_estatisticas_features.csv")
    by_split = describe_by_split(df)
    by_split.round(5).to_csv(config.RESULTS_DIR / "eda_estatisticas_por_split.csv", index=False)
    print(by_split.round(4).T.to_string())

    plot_series_and_splits(df, config.FIGURES_DIR / "eda_serie_e_splits.png")
    plot_price_distributions(df, config.FIGURES_DIR / "eda_distribuicao_preco_por_split.png")
    plot_return_distributions(df, config.FIGURES_DIR / "eda_distribuicao_log_retorno_por_split.png")
    plot_feature_histograms(df, config.FIGURES_DIR / "eda_histogramas_features.png")
    plot_acf(df, config.FIGURES_DIR / "eda_autocorrelacao.png")

    boundaries = boundary_table()
    boundaries.to_csv(config.RESULTS_DIR / "fronteiras_splits.csv", index=False)
    print(boundaries[(boundaries.janela == 4)].to_string())


if __name__ == "__main__":
    main()
