from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
import pandas as pd


@dataclass
class TrendAnalyzer:
    """
    Temporal trend analysis engine for breach data.

    Expects a normalized DataFrame with at least:
    - breach_date: datetime-like
    - individuals_affected: numeric
    - entity_name: string
    - breach_type: string
    - state: string
    """

    def _ensure_datetime(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        if "breach_date" not in out.columns:
            raise KeyError("Input DataFrame must contain a 'breach_date' column.")
        out["breach_date"] = pd.to_datetime(out["breach_date"], errors="coerce")
        if out["breach_date"].isna().all():
            raise ValueError("All values in 'breach_date' are NaT after parsing.")
        return out

    def compute_monthly_trends(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute monthly breach trends.

        - Resample by month-end on breach_date.
        - Metrics:
          * breach_count (count)
          * total_affected (sum of individuals_affected)
          * avg_affected (mean)
          * median_affected (median)
        - Adds:
          * period = YYYY-MM string
          * cumulative_count = running sum of breach_count
          * ma_12 = 12-month rolling mean of breach_count (min_periods=6)
        """
        df_dt = self._ensure_datetime(df)

        df_dt = df_dt.set_index("breach_date")

        # Use month-end frequency; "M" corresponds to calendar month-end.
        monthly = df_dt.resample("M").agg(
            breach_count=("individuals_affected", "size"),
            total_affected=("individuals_affected", "sum"),
            avg_affected=("individuals_affected", "mean"),
            median_affected=("individuals_affected", "median"),
        )

        monthly.index.name = "breach_date"
        monthly = monthly.sort_index()

        monthly["period"] = monthly.index.to_period("M").astype(str)
        monthly["cumulative_count"] = monthly["breach_count"].cumsum()
        monthly["ma_12"] = (
            monthly["breach_count"].rolling(window=12, min_periods=6).mean()
        )

        return monthly

    def compute_yearly_trends(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute yearly breach trends.

        - Group by year extracted from breach_date.
        - Metrics:
          * breach_count
          * total_affected
          * avg_affected
          * median_affected
          * unique_entities (nunique of entity_name)
          * unique_states (nunique of state)
        - Adds:
          * yoy_count_change (% change of breach_count vs previous year)
          * yoy_affected_change (% change of total_affected vs previous year)
        """
        df_dt = self._ensure_datetime(df)
        df_dt = df_dt.copy()
        df_dt["year"] = df_dt["breach_date"].dt.year

        grouped = df_dt.groupby("year")

        yearly = grouped.agg(
            breach_count=("breach_date", "size"),
            total_affected=("individuals_affected", "sum"),
            avg_affected=("individuals_affected", "mean"),
            median_affected=("individuals_affected", "median"),
            unique_entities=("entity_name", "nunique"),
            unique_states=("state", "nunique"),
        ).sort_index()

        yearly["yoy_count_change"] = yearly["breach_count"].pct_change() * 100.0
        yearly["yoy_affected_change"] = yearly["total_affected"].pct_change() * 100.0

        return yearly

    def compute_breach_type_trends(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute yearly breach-type composition trends.

        - Group by [year, breach_type], counting breaches.
        - Compute percentage share within each year.
        """
        df_dt = self._ensure_datetime(df)
        df_dt = df_dt.copy()
        df_dt["year"] = df_dt["breach_date"].dt.year

        grouped = (
            df_dt.groupby(["year", "breach_type"])
            .size()
            .reset_index(name="breach_count")
        )

        grouped["year_total"] = grouped.groupby("year")["breach_count"].transform("sum")
        grouped["percentage"] = np.where(
            grouped["year_total"] > 0,
            grouped["breach_count"] / grouped["year_total"] * 100.0,
            np.nan,
        )
        grouped = grouped.drop(columns=["year_total"])

        return grouped

    def detect_inflection_points(self, monthly: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Detect inflection points in the 12-month moving average (ma_12).

        - Computes the rate of change of ma_12.
        - Finds where the sign of the change reverses (up -> down or down -> up).
        - Returns a list of dicts with:
          * date: timestamp of inflection
          * breach_count_ma: ma_12 value at that point
          * direction: "up" if trend turns upward, "down" if turns downward
        """
        if "ma_12" not in monthly.columns:
            raise KeyError("monthly DataFrame must contain an 'ma_12' column.")

        ma = monthly["ma_12"]
        # Rate of change (first difference)
        delta = ma.diff()

        # Sign of change: -1 (down), 0 (flat), 1 (up)
        sign = np.sign(delta)

        inflections: List[Dict[str, Any]] = []
        # Iterate from second point onward to compare with prior sign
        for i in range(1, len(sign)):
            prev_sign = sign.iloc[i - 1]
            curr_sign = sign.iloc[i]

            # Ignore periods where either side is NaN or flat (0)
            if np.isnan(prev_sign) or np.isnan(curr_sign):
                continue
            if prev_sign == 0 or curr_sign == 0:
                continue

            if curr_sign != prev_sign:
                ts = monthly.index[i]
                value = ma.iloc[i]
                direction = "up" if curr_sign > 0 else "down"
                inflections.append(
                    {
                        "date": ts,
                        "breach_count_ma": float(value) if pd.notna(value) else np.nan,
                        "direction": direction,
                    }
                )

        return inflections

    def compute_headline_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Compute high-level headline metrics for a breach dataset.
        """
        df_dt = self._ensure_datetime(df)
        df_dt = df_dt.copy()
        df_dt["year"] = df_dt["breach_date"].dt.year

        total_breaches = int(len(df_dt))
        total_individuals_affected = int(
            df_dt["individuals_affected"].fillna(0).sum()
        )

        if df_dt["year"].isna().all():
            raise ValueError("All derived 'year' values are NaN; check 'breach_date' parsing.")

        current_year = int(df_dt["year"].max())

        yearly_counts = df_dt.groupby("year").size()
        current_year_breaches = int(yearly_counts.get(current_year, 0))
        previous_year = current_year - 1
        previous_year_breaches = int(yearly_counts.get(previous_year, 0))

        if previous_year_breaches > 0:
            yoy_change_pct = round(
                (current_year_breaches - previous_year_breaches)
                / previous_year_breaches
                * 100.0,
                1,
            )
        else:
            yoy_change_pct = None

        if "individuals_affected" in df_dt.columns and not df_dt["individuals_affected"].isna().all():
            idx_max = df_dt["individuals_affected"].idxmax()
            largest_breach_entity = df_dt.loc[idx_max, "entity_name"]
            largest_breach_count = int(df_dt.loc[idx_max, "individuals_affected"])
        else:
            largest_breach_entity = None
            largest_breach_count = 0

        most_common_breach_type = None
        if "breach_type" in df_dt.columns and not df_dt["breach_type"].isna().all():
            modes = df_dt["breach_type"].mode(dropna=True)
            if not modes.empty:
                most_common_breach_type = modes.iat[0]

        # Average breaches per month across the whole history
        year_month = df_dt["breach_date"].dt.to_period("M")
        num_months = int(year_month.nunique())
        if num_months > 0:
            avg_breaches_per_month = round(total_breaches / num_months, 1)
        else:
            avg_breaches_per_month = 0.0

        states_affected = (
            int(df_dt["state"].nunique()) if "state" in df_dt.columns else 0
        )

        return {
            "total_breaches": total_breaches,
            "total_individuals_affected": total_individuals_affected,
            "current_year": current_year,
            "current_year_breaches": current_year_breaches,
            "previous_year_breaches": previous_year_breaches,
            "yoy_change_pct": yoy_change_pct,
            "largest_breach_entity": largest_breach_entity,
            "largest_breach_count": largest_breach_count,
            "most_common_breach_type": most_common_breach_type,
            "avg_breaches_per_month": avg_breaches_per_month,
            "states_affected": states_affected,
        }

