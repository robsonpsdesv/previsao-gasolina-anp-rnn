"""Construção de janelas temporais (sequências) para treinamento supervisionado."""
import numpy as np


def create_sequences(X: np.ndarray, y: np.ndarray, window_size: int, horizon: int = 1):
    """Transforma séries 2D (amostras, features) em janelas 3D para RNNs.

    A janela [t-window_size+1, ..., t] prevê y no índice t (y já deve estar alinhado
    com o horizonte desejado, isto é, y[t] = valor real em t+horizon).

    Retorna:
        X_seq: shape (amostras, window_size, features)
        y_seq: shape (amostras, 1)
    """
    if len(X) != len(y):
        raise ValueError("X e y devem ter o mesmo número de linhas")
    if window_size < 1:
        raise ValueError("window_size deve ser >= 1")

    n_samples = len(X) - window_size + 1
    if n_samples <= 0:
        raise ValueError("window_size maior que o número de observações disponíveis")

    n_features = X.shape[1]
    X_seq = np.zeros((n_samples, window_size, n_features), dtype=X.dtype)
    y_seq = np.zeros((n_samples, 1), dtype=y.dtype)

    for i in range(n_samples):
        X_seq[i] = X[i:i + window_size]
        y_seq[i] = y[i + window_size - 1]

    return X_seq, y_seq
