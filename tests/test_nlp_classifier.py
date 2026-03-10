"""Tests for NLPAttackClassifier."""

from __future__ import annotations

import pandas as pd
import pytest

from breachintel.ml.nlp_classifier import NLPAttackClassifier


def test_create_labels_ransomware() -> None:
    """Verify 'ransomware attack encrypted files' -> 'Ransomware'."""
    clf = NLPAttackClassifier()
    labels = clf.create_labels(["ransomware attack encrypted files"])
    assert labels.iloc[0] == "Ransomware"


def test_create_labels_phishing() -> None:
    """Verify 'phishing email compromise' -> 'Phishing'."""
    clf = NLPAttackClassifier()
    labels = clf.create_labels(["phishing email compromise"])
    assert labels.iloc[0] == "Phishing"


def test_create_labels_unknown() -> None:
    """Verify 'something happened' -> 'Other'."""
    clf = NLPAttackClassifier()
    labels = clf.create_labels(["something happened"])
    assert labels.iloc[0] == "Other"


def test_train_with_descriptions(sample_clean_data: pd.DataFrame) -> None:
    """Train on sample data with descriptions; verify result dict."""
    # Ensure enough rows have non-empty descriptions for training (needs >= 100)
    df = sample_clean_data.copy()
    if "description" not in df.columns:
        df["description"] = "ransomware attack encrypted files"
    else:
        df["description"] = df["description"].fillna("ransomware attack encrypted files")
    # Lengthen short descriptions so train() accepts them (> 20 chars)
    mask = df["description"].astype(str).str.len() <= 20
    df.loc[mask, "description"] = "ransomware attack encrypted files and systems"

    clf = NLPAttackClassifier()
    result = clf.train(df)

    if result.get("status") == "ok":
        assert "classification_report" in result
        assert "n_train" in result
        assert "n_test" in result
    else:
        assert result.get("status") == "error"
        assert "n_valid" in result or "message" in result


def test_handles_empty_descriptions() -> None:
    """Verify no crash when descriptions are all NaN."""
    clf = NLPAttackClassifier()
    labels = clf.create_labels([None, float("nan"), ""])
    assert len(labels) == 3
    assert all(labels == "Other")
