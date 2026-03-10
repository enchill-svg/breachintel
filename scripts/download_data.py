from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

# Ensure project root is on sys.path so "breachintel" can be imported
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from breachintel.config import settings
from breachintel.data.collector import BreachDataCollector
from breachintel.data.cleaner import BreachDataCleaner
from breachintel.data.feature_engineer import FeatureEngineer
from breachintel.data.validator import validate_cleaned_data
from breachintel.utils.logger import logger


def main() -> None:
    logger.info("Starting BreachIntel data pipeline.")

    try:
        # 1) Collect and merge raw data
        collector = BreachDataCollector()
        merged_df = collector.load_and_merge()
        logger.info(f"Merged raw data shape: {merged_df.shape}")
        print(f"[collector] merged rows: {len(merged_df):,}")

        # 2) Clean merged data
        cleaner = BreachDataCleaner()
        cleaned_df = cleaner.clean()
        logger.info(f"Cleaned data shape: {cleaned_df.shape}")
        print(f"[cleaner] cleaned rows: {len(cleaned_df):,}")

        # 3) Validate cleaned data
        is_valid = validate_cleaned_data(cleaned_df)
        logger.info(f"Validation result: {is_valid}")
        print(f"[validator] validation passed: {is_valid}")

        # 4) Engineer tabular features
        fe = FeatureEngineer()
        features_df = fe.engineer_tabular_features(cleaned_df)

        processed_dir = Path(settings.data_processed_dir)
        processed_dir.mkdir(parents=True, exist_ok=True)
        features_path = processed_dir / "breaches_features.csv"
        features_df.to_csv(features_path, index=False)

        logger.info(
            f"Saved engineered features to {features_path} with shape {features_df.shape}."
        )
        print(
            f"[features] saved to {features_path}, "
            f"shape={features_df.shape}"
        )

        logger.info("BreachIntel data pipeline completed successfully.")
        print("Data pipeline completed successfully.")

    except Exception as exc:  # noqa: BLE001
        logger.exception(f"Data pipeline failed: {exc}")
        print(f"ERROR: Data pipeline failed - {exc}")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()

