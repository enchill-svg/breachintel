"""Shared pytest fixtures for BreachIntel test suite."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from breachintel.data.cleaner import BreachDataCleaner


RAW_COLUMNS = [
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

STATES = ["CA", "TX", "NY", "FL", "IL", "PA", "OH", "GA", "NC", "MI"]

ENTITY_TYPES = [
    "Healthcare Provider",
    "Health Plan",
    "Business Associate",
    "Healthcare Clearing House",
]
ENTITY_WEIGHTS = [0.6, 0.2, 0.15, 0.05]

BREACH_TYPES = [
    "Hacking/IT Incident",
    "Unauthorized Access/Disclosure",
    "Theft",
    "Loss",
    "Improper Disposal",
    "Other",
]
BREACH_WEIGHTS = [0.5, 0.2, 0.15, 0.1, 0.03, 0.02]

DESCRIPTION_KEYWORDS = [
    "ransomware attack encrypted files",
    "phishing email compromise",
    "unauthorized access by former employee",
    "stolen laptop from vehicle",
    "hacking incident",
    "network server compromised",
]


def _random_date_string(rng: np.random.Generator) -> str:
    """Random date between 2015-01-01 and 2025-12-31 as MM/DD/YYYY."""
    start = np.datetime64("2015-01-01")
    end = np.datetime64("2025-12-31")
    days = (end - start).astype(int)
    offset = rng.integers(0, days + 1)
    d = start + np.timedelta64(offset, "D")
    return pd.Timestamp(d).strftime("%m/%d/%Y")


def _make_description(rng: np.random.Generator, use_keyword: bool) -> str:
    if use_keyword:
        return rng.choice(DESCRIPTION_KEYWORDS)
    if rng.random() < 0.5:
        return "Breach of protected health information."
    return ""


@pytest.fixture
def sample_raw_data() -> pd.DataFrame:
    """Create a 500-row DataFrame mimicking raw merged data (after collector renames)."""
    rng = np.random.default_rng(42)
    n = 500

    entities = [f"Entity_{i}" for i in range(n)]
    states = rng.choice(STATES, size=n)
    entity_types = rng.choice(ENTITY_TYPES, size=n, p=ENTITY_WEIGHTS)
    individuals = rng.lognormal(mean=8, sigma=2, size=n).astype(int)
    individuals = np.maximum(individuals, 1)

    dates = [_random_date_string(rng) for _ in range(n)]
    breach_types = rng.choice(BREACH_TYPES, size=n, p=BREACH_WEIGHTS)
    locations = rng.choice(
        ["Network Server", "Email", "Paper/Films", "Portable Device", "Desktop", "EMR", "Other"],
        size=n,
    )
    ba_present = rng.choice(["Yes", "No"], size=n, p=[0.3, 0.7])

    use_keyword = rng.random(size=n) < 0.5
    descriptions = [_make_description(rng, u) for u in use_keyword]

    return pd.DataFrame(
        {
            "Name of Covered Entity": entities,
            "State": states,
            "Covered Entity Type": entity_types,
            "Individuals Affected": individuals,
            "Breach Submission Date": dates,
            "Type of Breach": breach_types,
            "Location of Breached Information": locations,
            "Business Associate Present": ba_present,
            "Web Description": descriptions,
        }
    )


@pytest.fixture
def sample_clean_data(sample_raw_data: pd.DataFrame) -> pd.DataFrame:
    """Run the cleaner's individual methods on sample_raw_data to produce clean data."""
    cleaner = BreachDataCleaner()
    df = sample_raw_data.copy()
    df = cleaner._normalize_columns(df)
    df = cleaner._parse_dates(df)
    df = cleaner._cast_numerics(df)
    df = cleaner._standardize_entity_types(df)
    df = cleaner._standardize_breach_types(df)
    df = cleaner._normalize_states(df)
    df = cleaner._standardize_locations(df)
    df = cleaner._handle_missing_values(df)
    df = cleaner._add_derived_columns(df)
    return df
