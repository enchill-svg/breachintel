from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import json

import pandas as pd

from breachintel.config import settings
from breachintel.data.feature_engineer import FeatureEngineer
from breachintel.ml.forecaster import BreachForecaster
from breachintel.ml.nlp_classifier import NLPAttackClassifier
from breachintel.ml.severity_model import SeverityModel
from breachintel.utils.logger import logger


def load_clean_data() -> pd.DataFrame:
    path = settings.data_processed_dir / "breaches_clean.csv"
    logger.info(f"Loading cleaned breach data from {path}")
    df = pd.read_csv(path, parse_dates=["breach_date"])
    logger.info(f"Loaded cleaned data with shape {df.shape}")
    return df


def load_or_create_features(clean_df: pd.DataFrame) -> pd.DataFrame:
    features_path = settings.data_processed_dir / "breaches_features.csv"
    if features_path.exists():
        logger.info(f"Loading existing feature matrix from {features_path}")
        features_df = pd.read_csv(features_path)
        logger.info(f"Loaded features with shape {features_df.shape}")
        return features_df

    logger.info("Feature matrix not found; running FeatureEngineer to generate features.")
    fe = FeatureEngineer()
    features_df = fe.engineer_tabular_features(clean_df)
    features_df.to_csv(features_path, index=False)
    logger.info(f"Saved engineered features to {features_path}")
    return features_df


def train_severity_model(features_df: pd.DataFrame) -> Dict[str, Any]:
    logger.info("Training SeverityModel...")
    severity_model = SeverityModel()
    metadata = severity_model.train(features_df)
    severity_model.save()
    logger.info("SeverityModel training completed and artifacts saved.")
    return metadata


def train_nlp_classifier(clean_df: pd.DataFrame) -> Dict[str, Any]:
    logger.info("Training NLPAttackClassifier...")

    # Filter to rows with non-empty descriptions; under-investigation rows have no descriptions.
    if "description" not in clean_df.columns:
        logger.warning(
            "Column 'description' missing from cleaned data; "
            "skipping NLPAttackClassifier training."
        )
        return {
            "status": "error",
            "message": "No 'description' column in cleaned data.",
            "n_valid": 0,
        }

    desc = clean_df["description"].astype(str)
    mask = desc.notna() & desc.str.strip().ne("")
    valid_df = clean_df.loc[mask].copy()

    logger.info(
        "Filtered breach descriptions for NLP training: "
        f"{len(valid_df)} non-empty descriptions (archive records only)."
    )

    nlp_clf = NLPAttackClassifier()
    result = nlp_clf.train(valid_df)

    if result.get("status") == "ok":
        nlp_clf.save()
        logger.info("NLPAttackClassifier training completed and pipeline saved.")
    else:
        logger.warning(
            "NLPAttackClassifier training did not complete successfully: "
            f"{result.get('message')}"
        )

    # Attach count of valid rows used
    result.setdefault("n_valid", int(len(valid_df)))
    return result


def train_forecaster(clean_df: pd.DataFrame) -> Dict[str, Any]:
    logger.info("Training BreachForecaster...")
    forecaster = BreachForecaster()
    forecast_df = forecaster.train_and_forecast(clean_df)
    forecaster.save()

    summary = forecaster.get_forecast_summary()
    logger.info(
        "BreachForecaster summary: "
        f"{summary['forecast_start']} -> {summary['forecast_end']}, "
        f"avg={summary['avg_predicted_monthly']}, "
        f"trend={summary['trend_direction']}"
    )

    # Optionally persist forecast for inspection
    forecast_path = settings.model_dir / "breach_forecast.csv"
    Path(settings.model_dir).mkdir(parents=True, exist_ok=True)
    forecast_df.to_csv(forecast_path, index=False)
    logger.info(f"Saved full breach forecast to {forecast_path}")

    return summary


def main() -> None:
    logger.info("=== Starting model training pipeline ===")

    clean_df = load_clean_data()
    features_df = load_or_create_features(clean_df)

    all_metadata: Dict[str, Any] = {}

    # 1) Severity model
    try:
        severity_meta = train_severity_model(features_df)
        all_metadata["severity_model"] = severity_meta
    except Exception as exc:  # noqa: BLE001
        logger.exception(f"SeverityModel training failed: {exc}")
        all_metadata["severity_model"] = {
            "status": "error",
            "message": str(exc),
        }

    # 2) NLP attack classifier
    try:
        nlp_meta = train_nlp_classifier(clean_df)
        all_metadata["nlp_attack_classifier"] = nlp_meta
    except Exception as exc:  # noqa: BLE001
        logger.exception(f"NLPAttackClassifier training failed: {exc}")
        all_metadata["nlp_attack_classifier"] = {
            "status": "error",
            "message": str(exc),
        }

    # 3) Time-series forecaster
    try:
        forecast_meta = train_forecaster(clean_df)
        all_metadata["breach_forecaster"] = forecast_meta
    except Exception as exc:  # noqa: BLE001
        logger.exception(f"BreachForecaster training failed: {exc}")
        all_metadata["breach_forecaster"] = {
            "status": "error",
            "message": str(exc),
        }

    # Print a simple summary to stdout
    logger.info("=== Model training summary ===")
    for name, meta in all_metadata.items():
        status = meta.get("status", "ok")
        logger.info(f"{name}: status={status}")

    # Save combined metadata JSON
    model_dir = Path(settings.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    meta_path = model_dir / "model_metadata.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(all_metadata, f, indent=2, default=str)

    logger.info(f"Saved combined model metadata to {meta_path}")
    logger.info("=== Model training pipeline finished ===")


if __name__ == "__main__":
    main()

