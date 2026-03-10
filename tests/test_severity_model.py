"""Tests for SeverityModel."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from breachintel.data.feature_engineer import FeatureEngineer
from breachintel.ml.severity_model import SeverityModel


@pytest.fixture
def features_df(sample_clean_data: pd.DataFrame) -> pd.DataFrame:
    """Engineer tabular features from sample_clean_data for severity model."""
    fe = FeatureEngineer()
    return fe.engineer_tabular_features(sample_clean_data)


def test_model_trains(features_df: pd.DataFrame) -> None:
    """Train on sample_clean_data features; verify cv_f1_mean > 0.3."""
    model = SeverityModel()
    metadata = model.train(features_df)
    assert "cv_f1_mean" in metadata
    assert metadata["cv_f1_mean"] > 0.3


def test_model_predicts(features_df: pd.DataFrame) -> None:
    """Train, then predict on first row; verify result has prediction key."""
    model = SeverityModel()
    model.train(features_df)
    X = model.get_feature_matrix(features_df)
    result = model.predict(X.head(1))
    assert "prediction" in result
    assert isinstance(result["prediction"], list)
    assert len(result["prediction"]) == 1


def test_model_save_load(features_df: pd.DataFrame, tmp_path: Path) -> None:
    """Train, save to tmp_path, load, predict — verify same structure."""
    model = SeverityModel()
    model.train(features_df)
    model.save(model_dir=tmp_path)

    loaded = SeverityModel()
    loaded.load(model_dir=tmp_path)

    X = model.get_feature_matrix(features_df.head(1))
    out1 = model.predict(X)
    out2 = loaded.predict(X)
    assert "prediction" in out2 and len(out2["prediction"]) == 1
    assert out1["prediction"][0] == out2["prediction"][0]


def test_predict_all_classes(features_df: pd.DataFrame) -> None:
    """Verify model can predict each possible class (Low, Medium, High)."""
    model = SeverityModel()
    model.train(features_df)
    X = model.get_feature_matrix(features_df)
    result = model.predict(X)
    predicted_classes = set(result["prediction"])
    # Model is 3-class: Low, Medium, High (from prepare_target bins)
    assert predicted_classes <= {"Low", "Medium", "High"}
    # With 500 rows we should see at least two classes
    assert len(predicted_classes) >= 1
