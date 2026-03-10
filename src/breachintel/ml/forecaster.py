from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from prophet import Prophet
from prophet.serialize import model_from_json, model_to_json

from ..config import settings
from ..utils.logger import logger


@dataclass
class BreachForecaster:
    """
    Prophet-based time-series forecaster for monthly breach counts.
    """

    forecast_months: int = 24
    model: Prophet = field(init=False)
    _history_df: Optional[pd.DataFrame] = field(default=None, init=False)
    _forecast_df: Optional[pd.DataFrame] = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            changepoint_prior_scale=0.1,
            seasonality_prior_scale=10.0,
            interval_width=0.90,
        )

    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare monthly breach counts in Prophet format.

        - Resample by month-end on breach_date.
        - Count rows as 'y'.
        - Rename breach_date index to 'ds'.
        """
        if "breach_date" not in df.columns:
            raise KeyError("Input DataFrame must contain 'breach_date' column.")

        df_dt = df.copy()
        df_dt["breach_date"] = pd.to_datetime(df_dt["breach_date"], errors="coerce")
        if df_dt["breach_date"].isna().all():
            raise ValueError("All values in 'breach_date' are NaT after parsing.")

        monthly = (
            df_dt.set_index("breach_date")
            .resample("ME")
            .size()
            .rename("y")
            .reset_index()
        )
        monthly = monthly.rename(columns={"breach_date": "ds"})
        self._history_df = monthly
        return monthly

    def train_and_forecast(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit the Prophet model and generate a forecast for forecast_months ahead.
        """
        history = self.prepare_data(df)
        if history.empty:
            raise ValueError("No historical data available to train forecaster.")

        self.model.fit(history)

        future = self.model.make_future_dataframe(
            periods=self.forecast_months,
            freq="ME",
        )
        forecast = self.model.predict(future)
        self._forecast_df = forecast

        logger.info(
            "Generated breach forecast: "
            f"start={history['ds'].min().date()}, "
            f"end={future['ds'].max().date()}, "
            f"forecast_months={self.forecast_months}"
        )

        return forecast

    def get_forecast_summary(self) -> Dict[str, Any]:
        """
        Summarize the forecasted period only.
        """
        if self._history_df is None or self._forecast_df is None:
            raise ValueError("Model has not been trained and forecast has not been generated yet.")

        history_end = self._history_df["ds"].max()
        forecast = self._forecast_df

        future_mask = forecast["ds"] > history_end
        future = forecast.loc[future_mask]
        if future.empty:
            raise ValueError("No future forecast rows found.")

        start = future["ds"].min()
        end = future["ds"].max()

        yhat = future["yhat"]
        avg_pred = round(float(yhat.mean()), 1)
        max_pred = round(float(yhat.max()), 1)
        min_pred = round(float(yhat.min()), 1)

        first_val = yhat.iloc[0]
        last_val = yhat.iloc[-1]
        trend_direction = "increasing" if last_val >= first_val else "decreasing"

        return {
            "forecast_start": start.isoformat(),
            "forecast_end": end.isoformat(),
            "avg_predicted_monthly": avg_pred,
            "max_predicted_monthly": max_pred,
            "min_predicted_monthly": min_pred,
            "trend_direction": trend_direction,
        }

    def save(self, model_dir: Optional[Path | str] = None) -> None:
        """
        Save Prophet model to disk using JSON serialization.
        """
        target_dir = Path(model_dir) if model_dir is not None else Path(settings.model_dir)
        target_dir.mkdir(parents=True, exist_ok=True)

        model_path = target_dir / "breach_forecaster.json"
        with model_path.open("w", encoding="utf-8") as f:
            f.write(model_to_json(self.model))

        logger.info(f"Saved BreachForecaster model to {model_path}")

    def load(self, model_dir: Optional[Path | str] = None) -> None:
        """
        Load Prophet model from disk.
        """
        target_dir = Path(model_dir) if model_dir is not None else Path(settings.model_dir)
        model_path = target_dir / "breach_forecaster.json"

        with model_path.open("r", encoding="utf-8") as f:
            self.model = model_from_json(f.read())

        logger.info(f"Loaded BreachForecaster model from {model_path}")

