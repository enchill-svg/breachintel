from __future__ import annotations

import math
from typing import Any, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from ..config import settings  # noqa: F401  # reserved for future path/config use
from ..utils.logger import logger


class FeatureEngineer:
    def __init__(self) -> None:
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=500,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=5,
            max_df=0.95,
        )

    def engineer_tabular_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create ML-ready tabular features from cleaned breach data.
        """
        logger.info("Engineering tabular ML features from cleaned breach data.")

        features = pd.DataFrame(index=df.index)

        # A) Temporal features
        for col in ["year", "quarter", "day_of_week"]:
            if col in df.columns:
                features[col] = df[col]

        if "breach_date" in df.columns:
            weekday = df["breach_date"].dt.weekday
            features["is_weekend"] = (weekday >= 5).astype(int)
        else:
            features["is_weekend"] = 0

        # B) Cyclical encoding of month
        if "month" in df.columns:
            month = df["month"].astype(float)
            features["month_sin"] = np.sin(2 * math.pi * month / 12.0)
            features["month_cos"] = np.cos(2 * math.pi * month / 12.0)
        else:
            features["month_sin"] = 0.0
            features["month_cos"] = 0.0

        # C) Categorical one-hot encodings
        cat_cols = {
            "entity_type": "entity",
            "breach_type": "breach",
            "breach_location": "location",
        }
        for col, prefix in cat_cols.items():
            if col in df.columns:
                dummies = pd.get_dummies(df[col].astype("category"), prefix=prefix)
                features = pd.concat([features, dummies], axis=1)

        # D) Historical state features
        if {"state", "breach_date", "individuals_affected"}.issubset(df.columns):
            logger.info("Computing historical state-level features.")
            tmp = df[["state", "breach_date", "individuals_affected"]].copy()
            tmp = tmp.sort_values(["state", "breach_date"])

            # cumulative prior breach count per state (exclude current row)
            tmp["state_prior_breach_count"] = (
                tmp.groupby("state").cumcount()
            )

            # mean breach size per state
            state_avg = (
                tmp.groupby("state")["individuals_affected"]
                .mean()
                .rename("state_avg_breach_size")
            )

            # map back to original index
            features["state_prior_breach_count"] = tmp["state_prior_breach_count"].reindex(
                df.index
            ).fillna(0).astype(int)
            features["state_avg_breach_size"] = df["state"].map(state_avg).fillna(0.0)
        else:
            features["state_prior_breach_count"] = 0
            features["state_avg_breach_size"] = 0.0

        # E) Business associate flag
        if "business_associate" in df.columns:
            ba = df["business_associate"].astype(str).str.lower()
            features["has_business_associate"] = ba.str.contains("yes").astype(int)
        else:
            features["has_business_associate"] = 0

        # F) Targets: severity and individuals_affected
        if "severity" in df.columns:
            features["target_severity"] = df["severity"]
        else:
            features["target_severity"] = pd.NA

        if "individuals_affected" in df.columns:
            features["individuals_affected"] = df["individuals_affected"]

        logger.info(f"Engineered tabular features with shape {features.shape}.")
        return features

    def engineer_nlp_features(
        self,
        df: pd.DataFrame,
    ) -> Tuple[Any, TfidfVectorizer, pd.Index]:
        """
        Engineer NLP features from the 'description' field using TF-IDF.

        Returns:
            (tfidf_matrix, fitted_vectorizer, valid_indices)
        """
        if "description" not in df.columns:
            logger.warning(
                "Column 'description' missing; cannot engineer NLP features. "
                "Returning empty matrix."
            )
            from scipy.sparse import csr_matrix  # local import to avoid hard dependency

            empty = csr_matrix((0, 0))
            return empty, self.tfidf_vectorizer, df.index[:0]

        # Filter rows with non-empty descriptions
        desc = df["description"].astype(str)
        mask = desc.notna() & desc.str.len().gt(10)
        valid_df = df.loc[mask].copy()

        logger.info(
            f"Engineering NLP features from descriptions: "
            f"{mask.sum()} rows with sufficient text."
        )

        if valid_df.empty:
            from scipy.sparse import csr_matrix

            empty = csr_matrix((0, 0))
            logger.warning("No valid descriptions found for NLP feature engineering.")
            return empty, self.tfidf_vectorizer, valid_df.index

        corpus = valid_df["description"].str.lower().tolist()
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(corpus)

        logger.info(
            f"Generated TF-IDF matrix with shape {tfidf_matrix.shape} "
            f"for {len(valid_df)} records."
        )

        return tfidf_matrix, self.tfidf_vectorizer, valid_df.index


if __name__ == "__main__":
    logger.info("FeatureEngineer module executed directly; no standalone action defined.")

