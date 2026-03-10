from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from ..config import settings
from ..utils.logger import logger


EXPECTED_COLUMNS = [
    "Name of Covered Entity",
    "State",
    "Covered Entity Type",
    "Individuals Affected",
    "Breach Submission Date",
    "Type of Breach",
    "Location of Breached Information",
    "Business Associate Present",
    "Web Description",
]


class BreachDataCollector:
    def __init__(self, raw_dir: Optional[Path] = None) -> None:
        self.raw_dir = Path(raw_dir) if raw_dir is not None else Path(settings.data_raw_dir)

    def _load_single(self, filename: str, source_label: str) -> pd.DataFrame:
        path = self.raw_dir / filename
        if not path.exists():
            logger.error(f"Raw file not found: {path}")
            raise FileNotFoundError(path)

        logger.info(f"Loading raw breach data from {path}")
        df = pd.read_csv(path)

        # Fix garbled JSF column names by position, not by name
        cols = list(df.columns)
        if len(cols) <= 7:
            logger.error(
                f"Unexpected column count in {path}: "
                f"expected at least 8 columns, got {len(cols)}"
            )
            raise ValueError(f"Unexpected column count in {path}")

        cols[0] = "Name of Covered Entity"
        cols[7] = "Business Associate Present"
        df.columns = cols

        df["source"] = source_label
        logger.info(f"Loaded {len(df):,} rows from {path}")
        return df

    def load_and_merge(self) -> pd.DataFrame:
        """
        Load archive and under-investigation breach datasets, normalize, and merge.
        """
        archive_df = self._load_single("archive.csv", "archive")
        under_inv_df = self._load_single("under_investigation.csv", "under_investigation")

        before_archive = len(archive_df)
        before_under = len(under_inv_df)

        combined = pd.concat([archive_df, under_inv_df], ignore_index=True)

        # Deduplicate on key, preferring archive records (loaded first)
        key_cols = [
            "Name of Covered Entity",
            "Breach Submission Date",
            "Individuals Affected",
        ]
        before_dedup = len(combined)
        combined = combined.drop_duplicates(subset=key_cols, keep="first")
        after_dedup = len(combined)
        duplicates_removed = before_dedup - after_dedup

        self._validate_raw(combined)

        output_path = self.raw_dir / "breaches_merged.csv"
        combined.to_csv(output_path, index=False)

        logger.info(
            "Merged breach datasets: "
            f"archive_rows={before_archive:,}, "
            f"under_investigation_rows={before_under:,}, "
            f"duplicates_removed={duplicates_removed:,}, "
            f"final_rows={after_dedup:,}, "
            f"output='{output_path}'"
        )

        return combined

    def _validate_raw(self, df: pd.DataFrame) -> None:
        missing_cols = [c for c in EXPECTED_COLUMNS if c not in df.columns]
        if missing_cols:
            logger.warning(f"Missing expected columns in merged data: {missing_cols}")

        row_count = len(df)
        if row_count <= 1000:
            logger.warning(
                f"Row count is lower than expected (>1000). Current row_count={row_count}"
            )

        if "Breach Submission Date" in df.columns:
            dates = pd.to_datetime(
                df["Breach Submission Date"],
                errors="coerce",
            )
            if dates.notna().any():
                min_year = dates.dt.year.min()
                max_year = dates.dt.year.max()
                if pd.notna(min_year) and pd.notna(max_year) and min_year == max_year:
                    logger.warning(
                        "Breach Submission Date does not span multiple years: "
                        f"min_year={min_year}, max_year={max_year}"
                    )
            else:
                logger.warning(
                    "Unable to parse any dates in 'Breach Submission Date' for validation."
                )
        else:
            logger.warning(
                "Column 'Breach Submission Date' missing; cannot validate date range."
            )


if __name__ == "__main__":
    collector = BreachDataCollector()
    collector.load_and_merge()

