"""Tests for RiskScorer."""

from __future__ import annotations

import pandas as pd
import pytest

from breachintel.ml.risk_scorer import RiskScorer


def test_score_returns_valid_structure(sample_clean_data: pd.DataFrame) -> None:
    """Verify dict has overall_score, risk_level, components."""
    scorer = RiskScorer(sample_clean_data)
    result = scorer.score(
        entity_type="Healthcare Provider",
        state="CA",
        breach_type="Hacking/IT Incident",
    )
    assert "overall_score" in result
    assert "risk_level" in result
    assert "components" in result
    assert "entity_score" in result["components"]
    assert "state_score" in result["components"]
    assert "trend_score" in result["components"]


def test_score_range(sample_clean_data: pd.DataFrame) -> None:
    """Verify overall_score is between 0 and 100."""
    scorer = RiskScorer(sample_clean_data)
    result = scorer.score(entity_type="Healthcare Provider", state="CA")
    score = result["overall_score"]
    assert 0 <= score <= 100


def test_risk_levels(sample_clean_data: pd.DataFrame) -> None:
    """Verify risk_level is one of Low / Moderate / High / Critical."""
    scorer = RiskScorer(sample_clean_data)
    result = scorer.score(entity_type="Health Plan", state="TX")
    valid_levels = {"Low", "Moderate", "High", "Critical"}
    assert result["risk_level"] in valid_levels
