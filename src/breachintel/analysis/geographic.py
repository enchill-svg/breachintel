from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd

from ..utils.constants import STATE_POPULATIONS


@dataclass
class GeographicAnalyzer:
    """
    Geographic breach analysis.

    Expects normalized columns:
    - breach_date: datetime-like
    - individuals_affected: numeric
    - entity_name: string
    - breach_type: string
    - state: two-letter code (e.g., 'CA')
    """

    def _ensure_datetime_and_state(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        if "breach_date" not in out.columns:
            raise KeyError("Input DataFrame must contain a 'breach_date' column.")
        if "state" not in out.columns:
            raise KeyError("Input DataFrame must contain a 'state' column.")

        out["breach_date"] = pd.to_datetime(out["breach_date"], errors="coerce")
        if out["breach_date"].isna().all():
            raise ValueError("All values in 'breach_date' are NaT after parsing.")

        out["state"] = out["state"].astype(str).str.upper()
        return out

    def compute_state_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Summarize breach activity by state.

        - Group by state and compute:
          * breach_count
          * total_affected
          * avg_affected
          * median_affected
          * max_affected
          * unique_entities
          * first_breach (min date)
          * latest_breach (max date)
        - Add:
          * population from STATE_POPULATIONS
          * breaches_per_100k = breach_count / population * 100000 (round 2)
          * affected_per_capita = total_affected / population (round 4)
          * dominant_breach_type = mode of breach_type per state
        - Sort by breach_count descending.
        """
        df_norm = self._ensure_datetime_and_state(df)

        grouped = df_norm.groupby("state")

        summary = grouped.agg(
            breach_count=("breach_date", "size"),
            total_affected=("individuals_affected", "sum"),
            avg_affected=("individuals_affected", "mean"),
            median_affected=("individuals_affected", "median"),
            max_affected=("individuals_affected", "max"),
            unique_entities=("entity_name", "nunique"),
            first_breach=("breach_date", "min"),
            latest_breach=("breach_date", "max"),
        )

        # Attach population
        pop_map: Dict[str, int] = STATE_POPULATIONS
        summary["population"] = summary.index.to_series().map(pop_map).astype("float")

        # Rates per population
        summary["breaches_per_100k"] = np.where(
            summary["population"] > 0,
            np.round(summary["breach_count"] / summary["population"] * 100_000, 2),
            np.nan,
        )
        summary["affected_per_capita"] = np.where(
            summary["population"] > 0,
            np.round(summary["total_affected"] / summary["population"], 4),
            np.nan,
        )

        # Dominant breach type per state
        if "breach_type" in df_norm.columns:
            dominant = (
                df_norm.groupby("state")["breach_type"]
                .agg(lambda s: s.mode(dropna=True).iat[0] if not s.mode(dropna=True).empty else None)
            )
            summary["dominant_breach_type"] = dominant
        else:
            summary["dominant_breach_type"] = None

        summary = summary.sort_values("breach_count", ascending=False)
        return summary

    def compute_regional_trends(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute breach trends by Census region and year.

        - Maps states to one of: Northeast, Midwest, South, West.
        - Groups by [region, year] and computes:
          * breach_count
          * total_affected
        """
        df_norm = self._ensure_datetime_and_state(df)
        df_norm = df_norm.copy()
        df_norm["year"] = df_norm["breach_date"].dt.year

        region_map: Dict[str, str] = {
            # Northeast
            "CT": "Northeast",
            "ME": "Northeast",
            "MA": "Northeast",
            "NH": "Northeast",
            "RI": "Northeast",
            "VT": "Northeast",
            "NJ": "Northeast",
            "NY": "Northeast",
            "PA": "Northeast",
            # Midwest
            "IL": "Midwest",
            "IN": "Midwest",
            "MI": "Midwest",
            "OH": "Midwest",
            "WI": "Midwest",
            "IA": "Midwest",
            "KS": "Midwest",
            "MN": "Midwest",
            "MO": "Midwest",
            "NE": "Midwest",
            "ND": "Midwest",
            "SD": "Midwest",
            # South
            "DE": "South",
            "FL": "South",
            "GA": "South",
            "MD": "South",
            "NC": "South",
            "SC": "South",
            "VA": "South",
            "DC": "South",
            "WV": "South",
            "AL": "South",
            "KY": "South",
            "MS": "South",
            "TN": "South",
            "AR": "South",
            "LA": "South",
            "OK": "South",
            "TX": "South",
            # West
            "AZ": "West",
            "CO": "West",
            "ID": "West",
            "MT": "West",
            "NV": "West",
            "NM": "West",
            "UT": "West",
            "WY": "West",
            "AK": "West",
            "CA": "West",
            "HI": "West",
            "OR": "West",
            "WA": "West",
        }

        df_norm["region"] = df_norm["state"].map(region_map)

        regional = (
            df_norm.dropna(subset=["region", "year"])
            .groupby(["region", "year"])
            .agg(
                breach_count=("breach_date", "size"),
                total_affected=("individuals_affected", "sum"),
            )
            .reset_index()
        )

        return regional

