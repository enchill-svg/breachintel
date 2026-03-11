from __future__ import annotations

from math import inf
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ..config import settings
from ..utils.constants import (
    BREACH_TYPE_MAP,
    ENTITY_TYPE_MAP,
    SEVERITY_BINS,
    SEVERITY_LABELS,
    STATE_ABBREVIATIONS,
)
from ..utils.logger import logger


class BreachDataCleaner:
    def __init__(self) -> None:
        self.quality_issues: List[Dict[str, Any]] = []
        self.raw_path = Path(settings.data_raw_dir) / "breaches_merged.csv"
        self.processed_dir = Path(settings.data_processed_dir)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def clean(self, input_path: Optional[Path] = None) -> pd.DataFrame:
        """
        Run the full cleaning pipeline on the merged raw data.
        """
        path = Path(input_path) if input_path is not None else self.raw_path
        if not path.exists():
            logger.error(f"Merged raw file not found: {path}")
            raise FileNotFoundError(path)

        logger.info(f"Loading merged raw breaches from {path}")
        df = pd.read_csv(path)

        # Execute cleaning steps in the specified order
        df = self._normalize_columns(df)
        df = self._parse_dates(df)
        df = self._cast_numerics(df)
        df = self._standardize_entity_types(df)
        df = self._standardize_breach_types(df)
        df = self._normalize_states(df)
        df = self._standardize_locations(df)
        df = self._handle_missing_values(df)
        df = self._remove_duplicates(df)
        df = self._add_derived_columns(df)
        df = self._flag_quality_issues(df)

        # Save outputs
        cleaned_path = self.processed_dir / "breaches_clean.csv"
        quality_path = self.processed_dir / "quality_report.csv"

        df.to_csv(cleaned_path, index=False)
        logger.info(f"Saved cleaned breaches to {cleaned_path} ({len(df):,} rows)")

        if self.quality_issues:
            quality_df = pd.DataFrame(self.quality_issues)
            quality_df.to_csv(quality_path, index=False)
            logger.info(
                f"Saved quality report with {len(self.quality_issues)} issues to {quality_path}"
            )
        else:
            logger.info("No quality issues recorded during cleaning.")

        return df

    def _normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize raw HHS column names to internal snake_case.

        NOTE: The HHS OCR export uses 'Breach Submission Date' for the timestamp field.
        In the cleaned dataset we standardize this to 'breach_date' and treat it as the
        canonical breach submission date throughout the application.
        """
        rename_map = {
            "Name of Covered Entity": "entity_name",
            "State": "state",
            "Covered Entity Type": "entity_type",
            "Individuals Affected": "individuals_affected",
            "Breach Submission Date": "breach_date",
            "Type of Breach": "breach_type",
            "Location of Breached Information": "breach_location",
            "Business Associate Present": "business_associate",
            "Web Description": "description",
        }
        logger.info("Normalizing column names to snake_case.")
        df = df.rename(columns=rename_map)

        missing = [k for k, v in rename_map.items() if v not in df.columns]
        if missing:
            issue = {
                "issue": "missing_expected_columns_after_rename",
                "columns": missing,
                "count": len(missing),
                "action": "logged_warning_only",
            }
            self.quality_issues.append(issue)
            logger.warning(f"Missing expected columns after rename: {missing}")

        return df

    def _parse_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        if "breach_date" not in df.columns:
            logger.warning("Column 'breach_date' missing; skipping date parsing.")
            return df

        logger.info("Parsing breach_date as datetime.")
        parsed = pd.to_datetime(
            df["breach_date"],
            format="mixed",
            dayfirst=False,
            errors="coerce",
        )
        unparsable = parsed.isna().sum()
        if unparsable > 0:
            issue = {
                "issue": "unparsable_breach_date",
                "count": int(unparsable),
                "action": "set_to_NaT_and_may_drop_later",
            }
            self.quality_issues.append(issue)
            logger.warning(f"Unparseable breach_date values: {unparsable}")

        df["breach_date"] = parsed
        return df

    def _cast_numerics(self, df: pd.DataFrame) -> pd.DataFrame:
        if "individuals_affected" not in df.columns:
            logger.warning("Column 'individuals_affected' missing; skipping numeric casting.")
            return df

        logger.info("Casting individuals_affected to numeric.")
        df["individuals_affected"] = pd.to_numeric(
            df["individuals_affected"],
            errors="coerce",
        )

        non_positive = (df["individuals_affected"] <= 0).sum()
        if non_positive > 0:
            issue = {
                "issue": "non_positive_individuals_affected",
                "count": int(non_positive),
                "action": "will_be_dropped_in_missing_value_handling",
            }
            self.quality_issues.append(issue)
            logger.warning(f"Non-positive individuals_affected values: {non_positive}")

        return df

    def _standardize_entity_types(self, df: pd.DataFrame) -> pd.DataFrame:
        if "entity_type" not in df.columns:
            logger.warning("Column 'entity_type' missing; skipping entity type standardization.")
            return df

        logger.info("Standardizing entity types.")
        df["entity_type_raw"] = df["entity_type"]

        # Normalize to upper-case so it matches ENTITY_TYPE_MAP keys
        normalized = (
            df["entity_type"]
            .astype(str)
            .str.strip()
            .str.upper()
        )
        mapped = normalized.map(ENTITY_TYPE_MAP)
        df["entity_type"] = mapped.fillna("Other")

        unmapped_count = mapped.isna().sum()
        if unmapped_count > 0:
            issue = {
                "issue": "unmapped_entity_type",
                "count": int(unmapped_count),
                "action": "set_to_Other",
            }
            self.quality_issues.append(issue)
            logger.warning(f"Unmapped entity types: {unmapped_count}")

        return df

    def _standardize_breach_types(self, df: pd.DataFrame) -> pd.DataFrame:
        if "breach_type" not in df.columns:
            logger.warning("Column 'breach_type' missing; skipping breach type standardization.")
            return df

        logger.info("Standardizing breach types (using first value when multi-valued).")
        df["breach_type_raw"] = df["breach_type"]

        first_type = (
            df["breach_type"]
            .astype(str)
            .str.split(",")
            .str[0]
            .str.strip()
            .str.upper()
        )

        mapped = first_type.map(BREACH_TYPE_MAP)
        df["breach_type"] = mapped.fillna("Other")

        unmapped_count = mapped.isna().sum()
        if unmapped_count > 0:
            issue = {
                "issue": "unmapped_breach_type",
                "count": int(unmapped_count),
                "action": "set_to_Other",
            }
            self.quality_issues.append(issue)
            logger.warning(f"Unmapped breach types: {unmapped_count}")

        return df

    def _normalize_states(self, df: pd.DataFrame) -> pd.DataFrame:
        if "state" not in df.columns:
            logger.warning("Column 'state' missing; skipping state normalization.")
            return df

        logger.info("Normalizing state values to USPS codes.")
        df["state_raw"] = df["state"]

        normalized = (
            df["state"]
            .astype(str)
            .str.strip()
            .str.upper()
        )
        mapped = normalized.map(STATE_ABBREVIATIONS)
        df["state"] = mapped

        unrecognized_mask = mapped.isna()
        unrecognized_count = int(unrecognized_mask.sum())
        if unrecognized_count > 0:
            unique_unrecognized = sorted(set(normalized[unrecognized_mask]))
            issue = {
                "issue": "unrecognized_state",
                "count": unrecognized_count,
                "values": unique_unrecognized,
                "action": "state_set_to_NaN",
            }
            self.quality_issues.append(issue)
            logger.warning(
                f"Unrecognized state values ({unrecognized_count} rows): "
                f"{unique_unrecognized}"
            )

        return df

    def _standardize_locations(self, df: pd.DataFrame) -> pd.DataFrame:
        if "breach_location" not in df.columns:
            logger.warning(
                "Column 'breach_location' missing; skipping breach location standardization."
            )
            return df

        logger.info("Standardizing breach locations to primary categories.")
        df["breach_location_raw"] = df["breach_location"]

        primary = (
            df["breach_location"]
            .astype(str)
            .str.split(",")
            .str[0]
            .str.strip()
        )

        def map_location(value: str) -> str:
            mapping = {
                "Network Server": "Network Server",
                "Email": "Email",
                "Paper/Films": "Paper/Films",
                "Laptop": "Portable Device",
                "Desktop Computer": "Desktop",
                "Other Portable Electronic Device": "Portable Device",
                "Electronic Medical Record": "EMR",
                "Other": "Other",
            }
            return mapping.get(value, "Other")

        df["breach_location"] = primary.map(map_location)
        return df

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Handling missing critical and categorical values.")

        before = len(df)
        critical_mask = df["individuals_affected"].isna() | df["breach_date"].isna()
        dropped = int(critical_mask.sum())

        if dropped > 0:
            issue = {
                "issue": "dropped_rows_missing_critical_fields",
                "count": dropped,
                "action": "dropped",
            }
            self.quality_issues.append(issue)
            logger.warning(
                f"Dropping {dropped} rows with missing individuals_affected or breach_date."
            )

        df = df.loc[~critical_mask].copy()
        after = len(df)
        logger.info(f"Rows after dropping missing critical fields: {after:,} (from {before:,})")

        # Fill categorical fields with "Unknown"
        for col in ["entity_type", "breach_type", "breach_location", "business_associate"]:
            if col in df.columns:
                df[col] = df[col].fillna("Unknown")

        # description is left as-is, NaN is acceptable
        return df

    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Removing duplicate records based on entity, state, date, and individuals_affected.")
        key_cols = ["entity_name", "state", "breach_date", "individuals_affected"]
        missing_keys = [c for c in key_cols if c not in df.columns]
        if missing_keys:
            logger.warning(
                f"Cannot de-duplicate on full key; missing columns: {missing_keys}. "
                "Skipping duplicate removal."
            )
            return df

        before = len(df)
        df = df.drop_duplicates(subset=key_cols, keep="first")
        after = len(df)
        removed = before - after

        if removed > 0:
            issue = {
                "issue": "dropped_duplicate_records",
                "count": int(removed),
                "action": "dropped",
            }
            self.quality_issues.append(issue)
            logger.info(f"Removed {removed} duplicate records.")

        return df

    def _add_derived_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        if "breach_date" not in df.columns:
            logger.warning("Column 'breach_date' missing; skipping derived date features.")
            return df

        logger.info("Adding derived temporal and severity features.")
        df["year"] = df["breach_date"].dt.year
        df["month"] = df["breach_date"].dt.month
        df["quarter"] = df["breach_date"].dt.to_period("Q").astype(str)
        df["day_of_week"] = df["breach_date"].dt.day_name()
        df["year_month"] = df["breach_date"].dt.to_period("M").astype(str)

        # Use configured severity bins/labels, which match the requested thresholds
        df["severity"] = pd.cut(
            df["individuals_affected"],
            bins=SEVERITY_BINS,
            labels=SEVERITY_LABELS,
            include_lowest=True,
            right=False,
        )

        df["log_individuals"] = np.log1p(df["individuals_affected"])
        return df

    def _flag_quality_issues(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Flagging quality-related indicators.")

        # is_estimated: round multiple of 1000 and > 1000
        estimated_mask = (df["individuals_affected"] > 1000) & (
            (df["individuals_affected"] % 1000) == 0
        )
        df["is_estimated"] = estimated_mask

        est_count = int(estimated_mask.sum())
        if est_count > 0:
            issue = {
                "issue": "estimated_individuals_affected",
                "count": est_count,
                "action": "flagged_is_estimated",
            }
            self.quality_issues.append(issue)
            logger.info(f"Flagged {est_count} rows as estimated individuals_affected.")

        # future_date flag (compare with tz-naive "today" to match breach_date)
        now = pd.Timestamp.now(tz=None).normalize()
        future_mask = df["breach_date"] > now
        df["quality_flag"] = np.where(future_mask, "future_date", "")

        future_count = int(future_mask.sum())
        if future_count > 0:
            issue = {
                "issue": "breach_date_in_future",
                "count": future_count,
                "action": "flagged_quality_flag_future_date",
            }
            self.quality_issues.append(issue)
            logger.warning(f"Found {future_count} breaches with breach_date in the future.")

        return df


if __name__ == "__main__":
    cleaner = BreachDataCleaner()
    cleaner.clean()

