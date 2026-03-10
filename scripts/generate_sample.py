from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
from sklearn.model_selection import train_test_split

# Ensure project root is on sys.path so "breachintel" can be imported
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from breachintel.config import settings
from breachintel.utils.logger import logger


def main() -> None:
    processed_path = Path(settings.data_processed_dir) / "breaches_clean.csv"
    sample_dir = Path(settings.data_sample_dir)
    sample_dir.mkdir(parents=True, exist_ok=True)
    sample_path = sample_dir / "breaches_sample.csv"

    if not processed_path.exists():
        msg = (
            f"Processed data not found at {processed_path}. "
            "Run scripts/download_data.py first to generate cleaned data."
        )
        logger.error(msg)
        print(f"ERROR: {msg}")
        raise SystemExit(1)

    logger.info(f"Loading cleaned breaches from {processed_path} to create sample.")
    df = pd.read_csv(processed_path)

    if "severity" not in df.columns:
        msg = "Column 'severity' missing from cleaned data; cannot stratify sample."
        logger.error(msg)
        print(f"ERROR: {msg}")
        raise SystemExit(1)

    total_rows = len(df)
    n_sample = min(500, total_rows)

    if n_sample == 0:
        msg = "Cleaned dataset is empty; cannot create sample."
        logger.error(msg)
        print(f"ERROR: {msg}")
        raise SystemExit(1)

    # Use train_test_split with stratify to get ~proportional representation by severity
    test_size = n_sample / total_rows
    logger.info(
        f"Creating stratified sample of {n_sample} rows "
        f"from {total_rows} cleaned records (test_size={test_size:.4f})."
    )

    _, sample_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df["severity"],
        random_state=42,
    )

    # If rounding causes off-by-one, adjust by sampling more/less without stratification
    if len(sample_df) > n_sample:
        sample_df = sample_df.sample(n=n_sample, random_state=42)
    elif len(sample_df) < n_sample:
        extra = df.drop(sample_df.index)
        if not extra.empty:
            needed = min(n_sample - len(sample_df), len(extra))
            sample_df = pd.concat(
                [sample_df, extra.sample(n=needed, random_state=42)],
                ignore_index=True,
            )

    sample_df.to_csv(sample_path, index=False)

    severity_dist = sample_df["severity"].value_counts(normalize=True).sort_index()
    logger.info(
        f"Saved stratified sample to {sample_path} with {len(sample_df)} rows. "
        f"Severity distribution:\n{severity_dist}"
    )
    print(f"Sample saved to {sample_path} ({len(sample_df)} rows).")
    print("Severity distribution:")
    print(severity_dist)


if __name__ == "__main__":
    main()

