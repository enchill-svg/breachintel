from __future__ import annotations

from typing import Any

import pandas as pd
import pandera as pa
from pandera import Check, Column, DataFrameSchema

from ..utils.logger import logger


BREACH_SCHEMA = DataFrameSchema(
    {
        "entity_name": Column(str, nullable=True),
        "state": Column(str, Check.str_length(2, 2), nullable=True),
        "entity_type": Column(
            str,
            Check.isin(
                [
                    "Healthcare Provider",
                    "Business Associate",
                    "Health Plan",
                    "Healthcare Clearing House",
                    "Unknown",
                    "Other",
                ]
            ),
        ),
        "individuals_affected": Column(float, Check.greater_than(0), nullable=False),
        "breach_date": Column("datetime64[ns]", nullable=False),
        "breach_type": Column(
            str,
            Check.isin(
                [
                    "Hacking/IT Incident",
                    "Unauthorized Access/Disclosure",
                    "Theft",
                    "Loss",
                    "Improper Disposal",
                    "Other",
                    "Unknown",
                ]
            ),
        ),
        "year": Column(pa.Int, Check.in_range(2009, 2030), coerce=True),
        "severity": Column(
            str,
            Check.isin(["Low", "Medium", "High", "Critical"]),
            nullable=True,
        ),
    },
    checks=[
        Check(lambda df: len(df) > 1000, error="Row count must be greater than 1000."),
        Check(
            lambda df: df["year"].nunique() > 5,
            error="Data must span more than 5 distinct years.",
        ),
    ],
)


def validate_cleaned_data(df: pd.DataFrame) -> bool:
    """
    Validate the cleaned breach dataset against the BREACH_SCHEMA.

    Returns True if valid, otherwise raises pa.errors.SchemaErrors.
    """
    logger.info("Validating cleaned breach data with pandera schema.")
    try:
        BREACH_SCHEMA.validate(df, lazy=True)
    except pa.errors.SchemaErrors as exc:  # type: ignore[attr-defined]
        logger.error(
            "Breach data validation failed. "
            f"Failure cases:\n{exc.failure_cases.head(20)}"
        )
        raise
    else:
        logger.info("Cleaned breach data passed schema validation.")
        return True

