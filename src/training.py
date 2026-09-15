"""Preparação de dados por janela e rotina de treinamento de um experimento.

Garante que o scaler é ajustado somente no treino e que as sequências (janelas)
respeitam a ordem cronológica: a série completa é escalada com os parâmetros do
treino e depois cortada por data para formar treino/validação/teste das sequências,
evitando tanto vazamento de escala quanto perda dos primeiros pontos de cada partição.
"""
import time
from dataclasses import asdict

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from . import config
from .models import ModelConfig, build_model
from .preprocessing import (
    TARGET_COLUMN,
    apply_scalers,
    build_processed_dataset,
    chronological_split,
    fit_scalers,
)
from .sequences import create_sequences
from .utils import set_seeds


def load_or_build_processed_df() -> pd.DataFrame:
    path = config.PROCESSED_DIR / "serie_semanal_nacional.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return build_processed_dataset()


def prepare_window_data(window_size: int, horizon: int = config.HORIZON):
    """Retorna sequências de treino/validação/teste e os scalers ajustados no treino."""
    df = load_or_build_processed_df()
    train_df, val_df, test_df = chronological_split(df)

    feature_scaler, target_scaler = fit_scalers(train_df)
    X_all, y_all = apply_scalers(df, feature_scaler, target_scaler)

    X_seq, y_seq = create_sequences(X_all, y_all, window_size, horizon)
    seq_dates = df.index[window_size - 1:]  # data-base t de cada janela (alvo em t+horizon)

    train_mask = seq_dates <= train_df.index.max()
    val_mask = (seq_dates > train_df.index.max()) & (seq_dates <= val_df.index.max())
    test_mask = seq_dates > val_df.index.max()

    splits = {
        "train": (X_seq[train_mask], y_seq[train_mask], seq_dates[train_mask]),
        "val": (X_seq[val_mask], y_seq[val_mask], seq_dates[val_mask]),
        "test": (X_seq[test_mask], y_seq[test_mask], seq_dates[test_mask]),
    }
    return splits, feature_scaler, target_scaler


def train_experiment(
    cell_type: str,
    window_size: int,
    units: int = 64,
    n_layers: int = 1,
    dropout: float = 0.0,
    recurrent_dropout: float = 0.0,
    l2: float = 0.0,
    learning_rate: float = config.LEARNING_RATE,
    batch_size: int = config.BATCH_SIZE,
    max_epochs: int = config.MAX_EPOCHS,
    patience: int = config.EARLY_STOPPING_PATIENCE,
    seed: int = config.SEED,
    verbose: int = 0,
):
    """Treina uma configuração e retorna métricas, histórico, tempo e o modelo treinado."""
    set_seeds(seed)

    splits, feature_scaler, target_scaler = prepare_window_data(window_size)
    X_train, y_train, dates_train = splits["train"]
    X_val, y_val, dates_val = splits["val"]
    X_test, y_test, dates_test = splits["test"]

    cfg = ModelConfig(
        cell_type=cell_type,
        units=units,
        n_layers=n_layers,
        dropout=dropout,
        recurrent_dropout=recurrent_dropout,
        l2=l2,
        learning_rate=learning_rate,
    )
    model = build_model(input_shape=(window_size, X_train.shape[-1]), cfg=cfg)

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=max(3, patience // 3), min_lr=1e-6),
    ]

    start = time.perf_counter()
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=max_epochs,
        batch_size=batch_size,
        shuffle=False,  # jamais embaralhar sequências temporais
        callbacks=callbacks,
        verbose=verbose,
    )
    elapsed = time.perf_counter() - start
    n_epochs_run = len(history.history["loss"])

    y_val_pred = model.predict(X_val, verbose=0)
    y_test_pred = model.predict(X_test, verbose=0)

    result = {
        "cell_type": cell_type,
        "window_size": window_size,
        "units": units,
        "n_layers": n_layers,
        "dropout": dropout,
        "recurrent_dropout": recurrent_dropout,
        "l2": l2,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "n_params": model.count_params(),
        "n_epochs_run": n_epochs_run,
        "train_time_seconds": elapsed,
        "train_time_per_epoch": elapsed / n_epochs_run if n_epochs_run else None,
    }

    artifacts = {
        "model": model,
        "history": history.history,
        "feature_scaler": feature_scaler,
        "target_scaler": target_scaler,
        "y_val": y_val,
        "y_val_pred": y_val_pred,
        "dates_val": dates_val,
        "y_test": y_test,
        "y_test_pred": y_test_pred,
        "dates_test": dates_test,
    }
    return result, artifacts


def persistence_baseline(window_size: int = 1):
    """Baseline de persistência: previsão de t+1 = valor observado em t."""
    df = load_or_build_processed_df()
    train_df, val_df, test_df = chronological_split(df)

    def _metrics_for(split_df):
        y_true = split_df[TARGET_COLUMN].values
        y_pred = split_df["gasolina_preco_medio"].values  # persistência: ŷ_{t+1} = y_t
        return y_true, y_pred

    return {
        "train": _metrics_for(train_df),
        "val": _metrics_for(val_df),
        "test": _metrics_for(test_df),
    }
