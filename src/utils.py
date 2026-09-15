"""Utilitários: reprodutibilidade e registro do ambiente de execução."""
import platform
import random

import numpy as np
import tensorflow as tf

from . import config


def set_seeds(seed: int = config.SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def environment_report() -> dict:
    gpus = tf.config.list_physical_devices("GPU")
    report = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "tensorflow_version": tf.__version__,
        "keras_version": tf.keras.__version__,
        "numpy_version": np.__version__,
        "gpu_devices": [d.name for d in gpus] if gpus else [],
    }
    return report


if __name__ == "__main__":
    set_seeds()
    for k, v in environment_report().items():
        print(f"{k}: {v}")
