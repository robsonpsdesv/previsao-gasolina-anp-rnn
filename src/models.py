"""Arquiteturas recorrentes (SimpleRNN, GRU, LSTM) com configuração comparável.

Todas as arquiteturas compartilham a mesma camada de saída (Dense(1), regressão
linear), o mesmo otimizador, loss e métricas, para que a comparação entre famílias
recorrentes seja controlada e justa (seção 21 do prompt).
"""
from dataclasses import dataclass, field

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers


@dataclass
class ModelConfig:
    cell_type: str  # "SimpleRNN", "GRU" ou "LSTM"
    units: int = 64
    n_layers: int = 1
    dropout: float = 0.0
    recurrent_dropout: float = 0.0
    l2: float = 0.0
    learning_rate: float = 1e-3


_CELL_MAP = {
    "SimpleRNN": layers.SimpleRNN,
    "GRU": layers.GRU,
    "LSTM": layers.LSTM,
}


def build_model(input_shape: tuple[int, int], cfg: ModelConfig) -> keras.Model:
    if cfg.cell_type not in _CELL_MAP:
        raise ValueError(f"cell_type inválido: {cfg.cell_type}")

    cell_cls = _CELL_MAP[cfg.cell_type]
    reg = regularizers.l2(cfg.l2) if cfg.l2 > 0 else None

    model = keras.Sequential(name=f"{cfg.cell_type}_{cfg.n_layers}L_{cfg.units}u")
    model.add(layers.Input(shape=input_shape))

    for layer_idx in range(cfg.n_layers):
        return_sequences = layer_idx < cfg.n_layers - 1
        model.add(
            cell_cls(
                cfg.units,
                return_sequences=return_sequences,
                dropout=cfg.dropout,
                recurrent_dropout=cfg.recurrent_dropout,
                kernel_regularizer=reg,
            )
        )

    model.add(layers.Dense(1))

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=cfg.learning_rate),
        loss="mse",
        metrics=["mae"],
    )
    return model
