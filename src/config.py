"""Configuração central do projeto: caminhos, seeds e criação dos diretórios.

Todo o dataset (raw/processed/intermediate/metadata/cache) reside em
<projeto>/datasets/anp_combustiveis/ (ignorado pelo git).
"""
import os
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# --- Dataset dentro do projeto ---------------------------------------------------
DATA_ROOT = Path(
    os.environ.get("ANP_DATA_ROOT", PROJECT_ROOT / "datasets" / "anp_combustiveis")
)

RAW_DIR = DATA_ROOT / "raw"
PROCESSED_DIR = DATA_ROOT / "processed"
INTERMEDIATE_DIR = DATA_ROOT / "intermediate"
METADATA_DIR = DATA_ROOT / "metadata"
CACHE_DIR = DATA_ROOT / "cache"

# --- Diretórios do projeto (código, notebooks, resultados, relatório) -----------
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
MODELS_DIR = RESULTS_DIR / "models"
REPORT_DIR = PROJECT_ROOT / "report"
EXPERIMENTS_CSV = RESULTS_DIR / "experiments.csv"

# --- Fontes oficiais dos dados ---------------------------------------------------
ANP_SOURCE_URLS = {
    "semanal_brasil_2004_2012": (
        "https://www.gov.br/anp/pt-br/assuntos/precos-e-defesa-da-concorrencia/"
        "precos/precos-revenda-e-de-distribuicao-combustiveis/shlp/2001-2012/"
        "semanal-brasil-2004-a-2012.xlsx"
    ),
    "semanal_brasil_desde_2013": (
        "https://www.gov.br/anp/pt-br/assuntos/precos-e-defesa-da-concorrencia/"
        "precos/precos-revenda-e-de-distribuicao-combustiveis/shlp/semanal/"
        "semanal-brasil-desde-2013.xlsx"
    ),
}

# --- Problema de aprendizado ------------------------------------------------------
TARGET_PRODUCT = "GASOLINA COMUM"
FEATURE_PRODUCTS = ["GASOLINA COMUM", "ETANOL HIDRATADO", "OLEO DIESEL"]
WINDOW_SIZES = [4, 12, 26]
HORIZON = 1
UNIT_OPTIONS = [32, 64, 128]
BATCH_SIZE = 32
MAX_EPOCHS = 100
LEARNING_RATE = 1e-3
EARLY_STOPPING_PATIENCE = 15
TRAIN_FRAC = 0.60
VAL_FRAC = 0.20
TEST_FRAC = 0.20

SEED = 42


def validate_data_dirs(min_free_gb: float = 1.0) -> None:
    """Cria os diretórios de dados/resultados e verifica o espaço livre em disco."""
    DATA_ROOT.mkdir(parents=True, exist_ok=True)

    total, _, free = shutil.disk_usage(DATA_ROOT)
    free_gb = free / (1024 ** 3)
    if free_gb < min_free_gb:
        raise RuntimeError(
            f"Espaço livre insuficiente em {DATA_ROOT}: {free_gb:.2f} GB "
            f"(mínimo exigido: {min_free_gb} GB)."
        )

    print(f"[config] DATA_ROOT = {DATA_ROOT}")
    print(f"[config] Espaço livre: {free_gb:.2f} GB de {total / (1024 ** 3):.2f} GB")

    for d in (RAW_DIR, PROCESSED_DIR, INTERMEDIATE_DIR, METADATA_DIR, CACHE_DIR):
        d.mkdir(parents=True, exist_ok=True)

    for d in (RESULTS_DIR, FIGURES_DIR, MODELS_DIR, REPORT_DIR):
        d.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    validate_data_dirs()
