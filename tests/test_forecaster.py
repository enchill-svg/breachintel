"""Tests for BreachForecaster."""

from __future__ import annotations

import pandas as pd
import pytest

# Skip forecaster tests if Prophet fails to initialize (e.g. stan_backend / env)
try:
    from breachintel.ml.forecaster import BreachForecaster
    BreachForecaster()  # fail here if Prophet has stan_backend etc. issues
    _FORECASTER_AVAILABLE = True
except Exception:
    _FORECASTER_AVAILABLE = False
    BreachForecaster = None  # type: ignore[misc, assignment]


@pytest.mark.skipif(not _FORECASTER_AVAILABLE, reason="Prophet not usable in this environment")
def test_prepare_data(sample_clean_data: pd.DataFrame) -> None:
    """Verify output has 'ds' and 'y' columns."""
    fc = BreachForecaster()
    prepared = fc.prepare_data(sample_clean_data)
    assert "ds" in prepared.columns
    assert "y" in prepared.columns


@pytest.mark.skipif(not _FORECASTER_AVAILABLE, reason="Prophet not usable in this environment")
def test_forecast_generates_future(sample_clean_data: pd.DataFrame) -> None:
    """Verify forecast extends beyond the data's last date."""
    fc = BreachForecaster()
    fc.train_and_forecast(sample_clean_data)
    last_history = fc._history_df["ds"].max()
    forecast_dates = fc._forecast_df["ds"]
    future = forecast_dates[forecast_dates > last_history]
    assert len(future) >= 1
    assert future.min() > last_history


@pytest.mark.skipif(not _FORECASTER_AVAILABLE, reason="Prophet not usable in this environment")
def test_forecast_summary(sample_clean_data: pd.DataFrame) -> None:
    """Verify summary dict has required keys."""
    fc = BreachForecaster()
    fc.train_and_forecast(sample_clean_data)
    summary = fc.get_forecast_summary()
    required = [
        "forecast_start",
        "forecast_end",
        "avg_predicted_monthly",
        "trend_direction",
    ]
    for key in required:
        assert key in summary, f"Missing key: {key}"
