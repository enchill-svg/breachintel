from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..config import settings
from .logger import logger


def load_data() -> pd.DataFrame:
    """
    Load breach data with simple caching/fallback semantics.

    1. Try raw under_investigation.csv
    2. Fallback to sample breaches_sample.csv if raw is missing
    """
    raw_under_investigation = Path(settings.data_raw_dir) / "under_investigation.csv"
    sample_path = Path(settings.data_sample_dir) / "breaches_sample.csv"

    if raw_under_investigation.exists():
        path = raw_under_investigation
        source = "raw_under_investigation"
    elif sample_path.exists():
        path = sample_path
        source = "sample"
    else:
        msg = (
            "No breach data found. Expected one of: "
            f"{raw_under_investigation} or {sample_path}"
        )
        logger.error(msg)
        raise FileNotFoundError(msg)

    logger.info(f"Loading breach data from {path} ({source})")
    df = pd.read_csv(path, parse_dates=["breach_date"])
    logger.info(f"Loaded {len(df):,} rows from {path}")
    return df

