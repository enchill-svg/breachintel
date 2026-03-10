from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class AttackVectorAnalyzer:
    """
    Analysis focused on breach types, locations, and severity.

    Expects normalized columns:
    - breach_type: string
    - individuals_affected: numeric
    - breach_location: string
    - severity: categorical/label
    - breach_date: datetime-like (for year derivation where needed)
    """

    def _ensure_year(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        if "breach_date" not in out.columns:
            raise KeyError("Input DataFrame must contain a 'breach_date' column.")
        out["breach_date"] = pd.to_datetime(out["breach_date"], errors="coerce")
        if out["breach_date"].isna().all():
            raise ValueError("All values in 'breach_date' are NaT after parsing.")
        out["year"] = out["breach_date"].dt.year
        return out

    def compute_vector_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Summarize breaches by attack vector (breach_type).

        - Group by breach_type and compute:
          * count
          * total_affected
          * avg_affected
          * median_affected
          * max_affected
        - Add:
          * percentage of total breaches
        - Sort by count descending.
        """
        if "breach_type" not in df.columns:
            raise KeyError("Input DataFrame must contain a 'breach_type' column.")

        grouped = df.groupby("breach_type").agg(
            count=("breach_type", "size"),
            total_affected=("individuals_affected", "sum"),
            avg_affected=("individuals_affected", "mean"),
            median_affected=("individuals_affected", "median"),
            max_affected=("individuals_affected", "max"),
        )

        total_count = grouped["count"].sum()
        grouped["percentage"] = np.where(
            total_count > 0,
            grouped["count"] / total_count * 100.0,
            np.nan,
        )

        grouped = grouped.sort_values("count", ascending=False)
        return grouped

    def compute_vector_evolution(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Track how breach types evolve over time.

        - Group by [year, breach_type] and count breaches.
        - Compute percentage share within each year.
        """
        df_year = self._ensure_year(df)

        counts = (
            df_year.groupby(["year", "breach_type"])
            .size()
            .reset_index(name="breach_count")
        )

        counts["year_total"] = counts.groupby("year")["breach_count"].transform("sum")
        counts["percentage"] = np.where(
            counts["year_total"] > 0,
            counts["breach_count"] / counts["year_total"] * 100.0,
            np.nan,
        )
        counts = counts.drop(columns=["year_total"])

        return counts

    def compute_location_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze breaches by location of breached information.

        - Group by breach_location and compute:
          * count
          * total_affected
        - Add percentage share of breaches.
        - Sort by count descending.
        """
        if "breach_location" not in df.columns:
            raise KeyError("Input DataFrame must contain a 'breach_location' column.")

        grouped = df.groupby("breach_location").agg(
            count=("breach_location", "size"),
            total_affected=("individuals_affected", "sum"),
        )

        total_count = grouped["count"].sum()
        grouped["percentage"] = np.where(
            total_count > 0,
            grouped["count"] / total_count * 100.0,
            np.nan,
        )

        grouped = grouped.sort_values("count", ascending=False)
        return grouped

    def compute_vector_severity_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Build a severity matrix of breach_type vs severity using a crosstab.

        - Returns a DataFrame with margins (row/column totals).
        """
        if "breach_type" not in df.columns:
            raise KeyError("Input DataFrame must contain a 'breach_type' column.")
        if "severity" not in df.columns:
            raise KeyError("Input DataFrame must contain a 'severity' column.")

        matrix = pd.crosstab(
            df["breach_type"],
            df["severity"],
            margins=True,
            margins_name="Total",
        )
        return matrix

