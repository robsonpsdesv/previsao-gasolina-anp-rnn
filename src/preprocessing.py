"""Pré-processamento: construção da série semanal nacional multivariada.

Pivota os produtos de interesse para formato largo (uma linha por semana),
cria a variável alvo (preço médio de revenda da gasolina em t+1), valida a
integridade da série (ordem temporal, ausência de NaN, frequência semanal) e
persiste a versão processada em BinEXT.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from . import config
from .data_loader import load_combined_raw

FEATURE_COLUMNS = [
    "gasolina_preco_medio",
    "gasolina_preco_minimo",
    "gasolina_preco_maximo",
    "gasolina_desvio_padrao",
    "gasolina_coef_variacao",
    "gasolina_num_postos",
    "etanol_preco_medio",
    "diesel_preco_medio",
]
TARGET_COLUMN = "target_gasolina_t_plus_1"


def build_weekly_national_series() -> pd.DataFrame:
    """Constrói a série semanal nacional com as features definidas em config.FEATURE_PRODUCTS."""
    raw = load_combined_raw()
    raw = raw[raw["PRODUTO"].isin(config.FEATURE_PRODUCTS)].copy()

    rename_map = {
        "GASOLINA COMUM": "gasolina",
        "ETANOL HIDRATADO": "etanol",
        "OLEO DIESEL": "diesel",
    }
    raw["produto_key"] = raw["PRODUTO"].map(rename_map)

    pieces = []
    for produto_key, group in raw.groupby("produto_key"):
        g = group.set_index("DATA INICIAL").sort_index()
        if produto_key == "gasolina":
            piece = pd.DataFrame({
                "gasolina_preco_medio": g["PRECO MEDIO REVENDA"],
                "gasolina_preco_minimo": g["PRECO MINIMO REVENDA"],
                "gasolina_preco_maximo": g["PRECO MAXIMO REVENDA"],
                "gasolina_desvio_padrao": g["DESVIO PADRAO REVENDA"],
                "gasolina_coef_variacao": g["COEF DE VARIACAO REVENDA"],
                "gasolina_num_postos": g["NUMERO DE POSTOS PESQUISADOS"],
            })
        elif produto_key == "etanol":
            piece = pd.DataFrame({"etanol_preco_medio": g["PRECO MEDIO REVENDA"]})
        elif produto_key == "diesel":
            piece = pd.DataFrame({"diesel_preco_medio": g["PRECO MEDIO REVENDA"]})
        else:
            continue
        pieces.append(piece)

    wide = pd.concat(pieces, axis=1, join="inner")
    wide = wide.sort_index()
    wide.index.name = "data_inicial"

    # Verificação de schema/consistência: colunas esperadas presentes
    missing_cols = set(FEATURE_COLUMNS) - set(wide.columns)
    if missing_cols:
        raise ValueError(f"Colunas de features ausentes após o pivot: {missing_cols}")

    return wide[FEATURE_COLUMNS]


def add_target(wide: pd.DataFrame, horizon: int = config.HORIZON) -> pd.DataFrame:
    df = wide.copy()
    df[TARGET_COLUMN] = df["gasolina_preco_medio"].shift(-horizon)
    df = df.dropna(subset=[TARGET_COLUMN])
    return df


def run_data_quality_checks(df: pd.DataFrame) -> None:
    """Verificações automáticas exigidas pela seção 41 do prompt."""
    assert df.index.is_monotonic_increasing, "Ordem temporal não é crescente"
    assert not df.isna().any().any(), "Existem valores ausentes antes do treinamento"
    assert not df.duplicated().any(), "Existem linhas duplicadas na série"

    deltas = df.index.to_series().diff().dropna()
    modal_gap = deltas.mode()[0]
    irregular = deltas[deltas != modal_gap]
    if len(irregular) > 0:
        print(
            f"[preprocessing] Aviso: {len(irregular)} lacunas na periodicidade semanal "
            f"(gap modal={modal_gap}); maiores gaps:\n{irregular.sort_values(ascending=False).head()}"
        )

    n_features = len(FEATURE_COLUMNS)
    assert df.shape[1] >= n_features, "Número de features menor que o esperado"
    print(f"[preprocessing] OK: {df.shape[0]} semanas, {n_features} features, sem NaN, sem duplicatas")


def chronological_split(df: pd.DataFrame, train_frac=config.TRAIN_FRAC, val_frac=config.VAL_FRAC):
    n = len(df)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    train = df.iloc[:n_train]
    val = df.iloc[n_train:n_train + n_val]
    test = df.iloc[n_train + n_val:]

    assert train.index.max() < val.index.min(), "Vazamento: treino sobrepõe validação"
    assert val.index.max() < test.index.min(), "Vazamento: validação sobrepõe teste"

    print(
        f"[preprocessing] Split cronológico -> treino: {train.index.min().date()} a {train.index.max().date()} "
        f"({len(train)} semanas); validação: {val.index.min().date()} a {val.index.max().date()} ({len(val)}); "
        f"teste: {test.index.min().date()} a {test.index.max().date()} ({len(test)})"
    )
    return train, val, test


def fit_scalers(train: pd.DataFrame):
    """Ajusta um StandardScaler para as features e outro para o alvo, usando SOMENTE o treino.

    StandardScaler foi escolhido (em vez de MinMaxScaler) porque os preços de combustíveis não
    têm limites fixos conhecidos a priori e podem crescer fora do intervalo observado no treino
    (ex.: inflação/choques de preço); a padronização por z-score lida melhor com esses casos do
    que uma normalização min-max, que fica limitada aos extremos vistos no treino.
    """
    feature_scaler = StandardScaler()
    feature_scaler.fit(train[FEATURE_COLUMNS].values)

    target_scaler = StandardScaler()
    target_scaler.fit(train[[TARGET_COLUMN]].values)

    return feature_scaler, target_scaler


def apply_scalers(df: pd.DataFrame, feature_scaler: StandardScaler, target_scaler: StandardScaler):
    X = feature_scaler.transform(df[FEATURE_COLUMNS].values)
    y = target_scaler.transform(df[[TARGET_COLUMN]].values)
    return X.astype(np.float32), y.astype(np.float32)


def build_processed_dataset(save: bool = True) -> pd.DataFrame:
    wide = build_weekly_national_series()
    df = add_target(wide)
    run_data_quality_checks(df)

    if save:
        target_path = config.PROCESSED_DIR / "serie_semanal_nacional.parquet"
        assert str(target_path.resolve()).startswith(str(config.EXTERNAL_DRIVE.resolve()))
        df.to_parquet(target_path)
        print(f"[preprocessing] Dataset processado salvo em {target_path}")
    return df


if __name__ == "__main__":
    df = build_processed_dataset()
    print(df.describe())
    train, val, test = chronological_split(df)
    fs, ts = fit_scalers(train)
    Xtr, ytr = apply_scalers(train, fs, ts)
    print("X train shape:", Xtr.shape, "y train shape:", ytr.shape)
