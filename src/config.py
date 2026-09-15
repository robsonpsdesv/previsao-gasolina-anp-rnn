"""Configuração central do projeto: caminhos, seeds e validação do HD externo BinEXT.

Regra obrigatória do trabalho: todo o dataset (raw/processed/intermediate/metadata/cache)
deve residir exclusivamente em /Volumes/BinEXT/datasets/anp_combustiveis/. Nada disso pode
cair no SSD interno. Se o BinEXT não estiver montado, a execução deve parar aqui.
"""
import os
import shutil
from pathlib import Path

# --- Unidade externa obrigatória -------------------------------------------------
EXTERNAL_DRIVE = Path("/Volumes/BinEXT")

DATA_ROOT = Path(
    os.environ.get("ANP_DATA_ROOT", "/Volumes/BinEXT/datasets/anp_combustiveis")
)

RAW_DIR = DATA_ROOT / "raw"
PROCESSED_DIR = DATA_ROOT / "processed"
INTERMEDIATE_DIR = DATA_ROOT / "intermediate"
METADATA_DIR = DATA_ROOT / "metadata"
CACHE_DIR = DATA_ROOT / "cache"

# --- Diretórios do projeto (código, notebooks, resultados, relatório) -----------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
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
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
TEST_FRAC = 0.15

SEED = 42


def validate_external_drive(min_free_gb: float = 1.0) -> None:
    """Garante que o BinEXT está montado, legível, gravável e com espaço livre.

    Interrompe a execução (RuntimeError) em vez de usar o SSD interno como fallback.
    """
    if not EXTERNAL_DRIVE.exists():
        raise RuntimeError(
            f"A unidade externa BinEXT não está montada em {EXTERNAL_DRIVE}. "
            "Conecte o HD externo antes de executar o pipeline."
        )

    if not os.access(EXTERNAL_DRIVE, os.R_OK):
        raise RuntimeError(f"Sem permissão de leitura em {EXTERNAL_DRIVE}.")

    if not os.access(EXTERNAL_DRIVE, os.W_OK):
        raise RuntimeError(f"Sem permissão de escrita em {EXTERNAL_DRIVE}.")

    if not str(DATA_ROOT.resolve()).startswith(str(EXTERNAL_DRIVE.resolve())):
        raise RuntimeError(
            f"DATA_ROOT ({DATA_ROOT}) não está localizado abaixo de {EXTERNAL_DRIVE}."
        )

    total, used, free = shutil.disk_usage(EXTERNAL_DRIVE)
    free_gb = free / (1024 ** 3)
    if free_gb < min_free_gb:
        raise RuntimeError(
            f"Espaço livre insuficiente em {EXTERNAL_DRIVE}: {free_gb:.2f} GB "
            f"(mínimo exigido: {min_free_gb} GB)."
        )

    print(f"[BinEXT] Unidade montada em {EXTERNAL_DRIVE}")
    print(f"[BinEXT] DATA_ROOT = {DATA_ROOT}")
    print(f"[BinEXT] Espaço livre: {free_gb:.2f} GB de {total / (1024 ** 3):.2f} GB")

    for d in (RAW_DIR, PROCESSED_DIR, INTERMEDIATE_DIR, METADATA_DIR, CACHE_DIR):
        d.mkdir(parents=True, exist_ok=True)

    for d in (RESULTS_DIR, FIGURES_DIR, MODELS_DIR, REPORT_DIR):
        d.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    validate_external_drive()
