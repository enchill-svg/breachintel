from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..config import settings
from .logger import logger


def load_data() -> pd.DataFrame:
    """
    Load breach data for interactive analysis.

    Preference order:
    1. Cleaned data: data/processed/breaches_clean.csv (output of BreachDataCleaner)
    2. Raw under_investigation.csv (with Breach Submission Date) as a fallback
    3. Sample breaches_sample.csv (for demos / tests)
    """
    processed_path = Path(settings.data_processed_dir) / "breaches_clean.csv"
    raw_under_investigation = Path(settings.data_raw_dir) / "under_investigation.csv"
    sample_path = Path(settings.data_sample_dir) / "breaches_sample.csv"

    if processed_path.exists():
        path = processed_path
        source = "processed_clean"
        logger.info(f"Loading cleaned breach data from {path} ({source})")
        df = pd.read_csv(path, parse_dates=["breach_date"])
        logger.info(f"Loaded {len(df):,} rows from {path}")
        return df

    if raw_under_investigation.exists():
        path = raw_under_investigation
        source = "raw_under_investigation"
        logger.info(f"Loading raw breach data from {path} ({source})")
        df = pd.read_csv(path)
        # Normalize column names so downstream code can rely on canonical fields
        if "State" in df.columns and "state" not in df.columns:
            df = df.rename(columns={"State": "state"})
        if "Breach Submission Date" in df.columns and "breach_date" not in df.columns:
            df = df.rename(columns={"Breach Submission Date": "breach_date"})
        # Normalize date column name if needed
        if "breach_date" in df.columns:
            df["breach_date"] = pd.to_datetime(df["breach_date"], errors="coerce")
        logger.info(f"Loaded {len(df):,} rows from {path}")
        return df

    if sample_path.exists():
        path = sample_path
        source = "sample"
        logger.info(f"Loading sample breach data from {path} ({source})")
        df = pd.read_csv(path)
        # Ensure sample data uses canonical column names expected by the app
        if "State" in df.columns and "state" not in df.columns:
            df = df.rename(columns={"State": "state"})
        if "Breach Submission Date" in df.columns and "breach_date" not in df.columns:
            df = df.rename(columns={"Breach Submission Date": "breach_date"})
        if "breach_date" in df.columns:
            df["breach_date"] = pd.to_datetime(df["breach_date"], errors="coerce")
        logger.info(f"Loaded {len(df):,} rows from {path}")
        return df

    msg = (
        "No breach data found. Expected one of: "
        f"{processed_path}, {raw_under_investigation}, or {sample_path}"
    )
    logger.error(msg)
    raise FileNotFoundError(msg)

