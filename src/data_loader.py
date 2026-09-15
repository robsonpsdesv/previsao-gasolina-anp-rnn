"""Carregamento dos dados brutos da ANP diretamente do BinEXT.

Faz o download (se necessário, direto para RAW_DIR) e a leitura resiliente das
planilhas semanais nacionais, cujo schema muda ligeiramente entre os arquivos
histórico (2004-2012) e atual (desde 2013): linha de cabeçalho em posição
diferente e nomes de produto com/sem acentuação.
"""
import unicodedata
import urllib.request
from pathlib import Path

import pandas as pd

from . import config


def _strip_accents(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(c for c in normalized if not unicodedata.combining(c))


def ensure_raw_files_downloaded() -> list[Path]:
    """Garante que os arquivos brutos existam em RAW_DIR, baixando somente se necessário."""
    config.validate_external_drive()
    paths = []
    for name, url in config.ANP_SOURCE_URLS.items():
        filename = url.rsplit("/", 1)[-1]
        target = config.RAW_DIR / filename
        if target.exists() and target.stat().st_size > 0:
            print(f"[data_loader] Arquivo já existente, reutilizando: {target}")
        else:
            print(f"[data_loader] Baixando {url} -> {target}")
            urllib.request.urlretrieve(url, target)
            if not target.exists() or target.stat().st_size == 0:
                raise RuntimeError(f"Falha ao baixar {url} para {target}")
        paths.append(target)
    return paths


def _find_header_row(path: Path, sheet_name: str, max_scan: int = 30) -> int:
    """Localiza a linha de cabeçalho procurando a coluna 'DATA INICIAL'."""
    raw = pd.read_excel(path, sheet_name=sheet_name, header=None, nrows=max_scan)
    for i in range(len(raw)):
        row = raw.iloc[i].astype(str).str.upper()
        if row.str.contains("DATA INICIAL").any():
            return i
    raise ValueError(f"Não foi possível localizar o cabeçalho em {path} ({sheet_name})")


def _normalize_columns(columns: list[str]) -> list[str]:
    return [_strip_accents(str(c)).strip().upper() for c in columns]


def load_raw_file(path: Path) -> pd.DataFrame:
    """Lê um arquivo bruto da ANP, detectando o cabeçalho automaticamente."""
    xls = pd.ExcelFile(path)
    sheet_name = xls.sheet_names[0]
    header_row = _find_header_row(path, sheet_name)
    df = pd.read_excel(path, sheet_name=sheet_name, header=header_row)
    df.columns = _normalize_columns(df.columns.tolist())
    df["PRODUTO"] = df["PRODUTO"].astype(str).map(_strip_accents).str.strip().str.upper()
    df["DATA INICIAL"] = pd.to_datetime(df["DATA INICIAL"])
    df["DATA FINAL"] = pd.to_datetime(df["DATA FINAL"])
    df["__source_file__"] = path.name
    return df


def load_combined_raw() -> pd.DataFrame:
    """Carrega e concatena todos os arquivos brutos, removendo duplicidades de semanas
    que aparecem em mais de um arquivo (fronteira entre 2012 e 2013)."""
    raw_paths = ensure_raw_files_downloaded()
    frames = [load_raw_file(p) for p in raw_paths]
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values(["DATA INICIAL", "PRODUTO"]).reset_index(drop=True)
    before = len(combined)
    combined = combined.drop_duplicates(subset=["DATA INICIAL", "DATA FINAL", "PRODUTO"], keep="first")
    after = len(combined)
    if before != after:
        print(f"[data_loader] Removidas {before - after} linhas duplicadas na fronteira dos arquivos")
    return combined


if __name__ == "__main__":
    df = load_combined_raw()
    print(df.shape)
    print(df["PRODUTO"].value_counts())
    print(df["DATA INICIAL"].min(), df["DATA INICIAL"].max())
