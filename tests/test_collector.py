"""Tests for BreachDataCollector."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from breachintel.data.collector import BreachDataCollector, EXPECTED_COLUMNS


@pytest.fixture
def raw_dir_with_merged_files(tmp_path: Path, sample_raw_data: pd.DataFrame) -> Path:
    """Create archive.csv and under_investigation.csv in tmp_path for load_and_merge."""
    archive = sample_raw_data.head(300).copy()
    under_inv = sample_raw_data.tail(200).copy()
    archive.to_csv(tmp_path / "archive.csv", index=False)
    under_inv.to_csv(tmp_path / "under_investigation.csv", index=False)
    return tmp_path


def test_load_and_merge_column_rename(raw_dir_with_merged_files: Path) -> None:
    """After load_and_merge, columns are correctly named (no garbled/javax names)."""
    collector = BreachDataCollector(raw_dir=raw_dir_with_merged_files)
    result = collector.load_and_merge()

    for col in result.columns:
        assert "javax" not in col.lower(), f"Column should not contain 'javax': {col}"
        assert col.strip() == col, f"Column should have no leading/trailing space: {col}"

    for expected in EXPECTED_COLUMNS:
        assert expected in result.columns, f"Expected column missing: {expected}"
    assert "source" in result.columns


def test_load_and_merge_row_count(raw_dir_with_merged_files: Path) -> None:
    """Combined rows are reasonable (>0)."""
    collector = BreachDataCollector(raw_dir=raw_dir_with_merged_files)
    result = collector.load_and_merge()
    assert len(result) > 0
    assert len(result) <= 500  # archive 300 + under_inv 200, minus possible dedup


def test_source_column_exists(raw_dir_with_merged_files: Path) -> None:
    """Verify 'source' column with 'archive' and 'under_investigation' values."""
    collector = BreachDataCollector(raw_dir=raw_dir_with_merged_files)
    result = collector.load_and_merge()
    assert "source" in result.columns
    sources = set(result["source"].dropna().unique())
    assert "archive" in sources
    assert "under_investigation" in sources
