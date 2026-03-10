"""Tests for BreachDataCleaner."""

from __future__ import annotations

import pandas as pd
import pytest

from breachintel.data.cleaner import BreachDataCleaner


def test_normalize_columns(sample_raw_data: pd.DataFrame) -> None:
    """Verify snake_case column names after _normalize_columns."""
    cleaner = BreachDataCleaner()
    df = cleaner._normalize_columns(sample_raw_data.copy())
    for col in df.columns:
        assert col == col.lower().replace(" ", "_") or col.isidentifier(), (
            f"Expected snake_case or identifier: {col}"
        )
    assert "entity_name" in df.columns
    assert "breach_date" in df.columns
    assert "individuals_affected" in df.columns


def test_parse_dates(sample_raw_data: pd.DataFrame) -> None:
    """Verify breach_date is datetime after _parse_dates."""
    cleaner = BreachDataCleaner()
    df = cleaner._normalize_columns(sample_raw_data.copy())
    df = cleaner._parse_dates(df)
    assert pd.api.types.is_datetime64_any_dtype(df["breach_date"])


def test_handles_invalid_dates() -> None:
    """Create df with bad date; verify it becomes NaT, no crash."""
    cleaner = BreachDataCleaner()
    df = pd.DataFrame(
        {
            "entity_name": ["A"],
            "state": ["CA"],
            "entity_type": ["Healthcare Provider"],
            "individuals_affected": [1000],
            "breach_date": ["not-a-date"],
            "breach_type": ["Hacking/IT Incident"],
            "breach_location": ["Network Server"],
            "business_associate": ["No"],
            "description": [""],
        }
    )
    df = cleaner._parse_dates(df)
    assert df["breach_date"].isna().all()


def test_standardize_breach_types_multi_value() -> None:
    """Verify 'Theft, Unauthorized Access/Disclosure' becomes 'Theft' (first value)."""
    cleaner = BreachDataCleaner()
    df = pd.DataFrame(
        {
            "entity_name": ["E1"],
            "state": ["CA"],
            "entity_type": ["Healthcare Provider"],
            "individuals_affected": [500],
            "breach_date": ["01/15/2020"],
            "breach_type": ["Theft, Unauthorized Access/Disclosure"],
            "breach_location": ["Network Server"],
            "business_associate": ["No"],
            "description": [""],
        }
    )
    df = cleaner._normalize_columns(df)
    df = cleaner._parse_dates(df)
    df = cleaner._cast_numerics(df)
    df = cleaner._standardize_entity_types(df)
    df = cleaner._standardize_breach_types(df)
    assert df["breach_type"].iloc[0] == "Theft"


def test_severity_bins(sample_clean_data: pd.DataFrame) -> None:
    """Verify severity column has only valid values."""
    valid = {"Low", "Medium", "High", "Critical"}
    assert "severity" in sample_clean_data.columns
    unique = set(sample_clean_data["severity"].dropna().astype(str).unique())
    assert unique <= valid, f"Unexpected severity values: {unique - valid}"


def test_no_data_loss(sample_raw_data: pd.DataFrame, sample_clean_data: pd.DataFrame) -> None:
    """Verify cleaning does not lose more than 5% of valid rows."""
    before = len(sample_raw_data)
    after = len(sample_clean_data)
    assert after >= 0.95 * before, (
        f"Lost more than 5%: before={before}, after={after}"
    )


def test_derived_columns_exist(sample_clean_data: pd.DataFrame) -> None:
    """Verify year, month, quarter, severity, log_individuals all exist."""
    required = ["year", "month", "quarter", "severity", "log_individuals"]
    for col in required:
        assert col in sample_clean_data.columns, f"Missing derived column: {col}"
