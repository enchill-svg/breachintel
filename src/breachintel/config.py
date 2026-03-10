from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _env_path(name: str, default: Path) -> Path:
    value = os.getenv(name)
    return Path(value) if value else default


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    project_root: Path = PROJECT_ROOT
    data_raw_dir: Path = _env_path("DATA_RAW_DIR", PROJECT_ROOT / "data" / "raw")
    data_processed_dir: Path = _env_path("DATA_PROCESSED_DIR", PROJECT_ROOT / "data" / "processed")
    data_sample_dir: Path = PROJECT_ROOT / "data" / "sample"
    model_dir: Path = _env_path("MODEL_DIR", PROJECT_ROOT / "models")
    hhs_breach_url: str = os.getenv(
        "HHS_BREACH_URL",
        "https://ocrportal.hhs.gov/ocr/breach/breach_report.jsf",
    )
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    log_file: Path = _env_path("LOG_FILE", PROJECT_ROOT / "logs" / "breachintel.log")
    rf_n_estimators: int = _env_int("RF_N_ESTIMATORS", 200)
    rf_max_depth: int = _env_int("RF_MAX_DEPTH", 15)
    rf_random_state: int = 42
    test_size: float = 0.2
    census_api_key: str = os.getenv("CENSUS_API_KEY", "")


settings = Settings()

