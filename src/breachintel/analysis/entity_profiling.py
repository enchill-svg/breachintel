from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class EntityProfiler:
    """
    Entity-level breach profiling.

    Expects normalized columns:
    - entity_name: string
    - entity_type: string
    - state: string
    - individuals_affected: numeric
    - breach_date: datetime-like
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

    def compute_entity_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Summarize breaches by entity_type.

        - Group by entity_type and compute:
          * breach_count
          * total_affected
          * avg_affected
          * median_affected
          * max_affected
          * unique_entities
        - Add:
          * pct_of_breaches
          * repeat_offenders: count of entities with >1 breach per type
        """
        if "entity_type" not in df.columns:
            raise KeyError("Input DataFrame must contain an 'entity_type' column.")
        if "entity_name" not in df.columns:
            raise KeyError("Input DataFrame must contain an 'entity_name' column.")

        grouped = df.groupby("entity_type")

        summary = grouped.agg(
            breach_count=("entity_type", "size"),
            total_affected=("individuals_affected", "sum"),
            avg_affected=("individuals_affected", "mean"),
            median_affected=("individuals_affected", "median"),
            max_affected=("individuals_affected", "max"),
            unique_entities=("entity_name", "nunique"),
        )

        total_breaches = summary["breach_count"].sum()
        summary["pct_of_breaches"] = np.where(
            total_breaches > 0,
            summary["breach_count"] / total_breaches * 100.0,
            np.nan,
        )

        # Repeat offenders: entities with >1 breach within each entity_type
        counts_by_entity = (
            df.groupby(["entity_type", "entity_name"])
            .size()
            .reset_index(name="breach_count")
        )
        repeat = (
            counts_by_entity[counts_by_entity["breach_count"] > 1]
            .groupby("entity_type")["entity_name"]
            .nunique()
        )
        summary["repeat_offenders"] = repeat
        summary["repeat_offenders"] = summary["repeat_offenders"].fillna(0).astype(int)

        return summary

    def find_most_breached_entities(
        self, df: pd.DataFrame, top_n: int = 20
    ) -> pd.DataFrame:
        """
        Identify the most breached entities overall.

        - Group by entity_name and compute:
          * breach_count
          * total_affected
          * entity_type (first)
          * state (first)
          * first_breach
          * latest_breach
        - Sort by breach_count descending and return top_n rows.
        """
        df_year = self._ensure_year(df)

        grouped = df_year.groupby("entity_name").agg(
            breach_count=("entity_name", "size"),
            total_affected=("individuals_affected", "sum"),
            entity_type=("entity_type", "first"),
            state=("state", "first"),
            first_breach=("breach_date", "min"),
            latest_breach=("breach_date", "max"),
        )

        top = grouped.sort_values(
            ["breach_count", "total_affected"], ascending=[False, False]
        ).head(top_n)

        return top

    def compute_entity_trend(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute breach trends by entity_type over time.

        - Group by [year, entity_type] and count breaches.
        - Compute percentage share within each year.
        """
        df_year = self._ensure_year(df)

        counts = (
            df_year.groupby(["year", "entity_type"])
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

