"""
preprocessing.py
Data loading and preprocessing utilities for SmartGrid-AI project.
"""

import os
import pandas as pd


def ensure_dirs():
    os.makedirs("artifacts", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    return True


def load_spain_energy(path="data/spain_energy.csv") -> pd.DataFrame:
    """Load and preprocess Spain energy dataset (handles tab/comma, trims cols)."""
    try:
        df = pd.read_csv(path, sep="\t", encoding="utf-8")
    except Exception:
        df = pd.read_csv(path, sep=",", encoding="utf-8")

    df.columns = df.columns.str.strip()
    print("Detected Spain columns:", df.columns.tolist())

    # Typical columns seen earlier: 'Hora','Real','Prevista','Programada'
    # Map into unified names if present.
    rename_map = {}
    if "Hora" in df.columns:
        rename_map["Hora"] = "datetime"
    if "Real" in df.columns:
        rename_map["Real"] = "real_demand"
    if "Prevista" in df.columns:
        rename_map["Prevista"] = "forecast_demand"
    if "Programada" in df.columns:
        rename_map["Programada"] = "scheduled_demand"

    if rename_map:
        df = df.rename(columns=rename_map)

    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], dayfirst=True, errors="coerce")
        df = df.sort_values("datetime").reset_index(drop=True)

        # Drop obvious duplicates / NaTs
        df = df.dropna(subset=["datetime"])
        df = df.drop_duplicates(subset=["datetime"])

    # Ensure numeric for demand columns when present
    for c in ("real_demand", "forecast_demand", "scheduled_demand"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


def load_stability(path="data/smartgrid_stability.csv") -> pd.DataFrame:
    """
    Load smart grid stability dataset (tab-separated).
    Normalizes column names to lower-case; expects 'stabf' as label.
    """
    df = pd.read_csv(path, sep="\t")
    df.columns = df.columns.str.strip().str.lower()
    # Some variants have 'stab' numeric + 'stabf' categorical; we keep 'stabf'
    if "stabf" not in df.columns:
        raise ValueError("Expected 'stabf' column (stable/unstable) in stability dataset.")
    return df


if __name__ == "__main__":
    ensure_dirs()
    spain = load_spain_energy()
    stability = load_stability()

    print("Spain dataset:", spain.shape)
    print(spain.head(), "\n")

    print("Stability dataset:", stability.shape)
    print(stability.head(), "\n")
