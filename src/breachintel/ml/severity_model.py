from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder

from ..config import settings
from ..utils.logger import logger


@dataclass
class SeverityModel:
    """
    Random Forest classifier for breach severity prediction.
    """

    model: RandomForestClassifier = field(init=False)
    label_encoder: LabelEncoder = field(init=False)
    feature_columns: Optional[List[str]] = field(default=None)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.model = RandomForestClassifier(
            n_estimators=settings.rf_n_estimators,
            max_depth=settings.rf_max_depth,
            class_weight="balanced",
            random_state=settings.rf_random_state,
            n_jobs=-1,
            min_samples_leaf=5,
        )
        self.label_encoder = LabelEncoder()

    def prepare_target(self, df: pd.DataFrame) -> pd.Series:
        """
        Prepare 3-class severity target based on individuals_affected.

        Bins:
        - [0, 1_000) -> "Low"
        - [1_000, 100_000) -> "Medium"
        - [100_000, inf) -> "High"
        """
        if "individuals_affected" not in df.columns:
            raise KeyError("Input DataFrame must contain 'individuals_affected' column.")

        values = df["individuals_affected"].astype(float)
        bins = [0.0, 1_000.0, 100_000.0, float("inf")]
        labels = ["Low", "Medium", "High"]
        target = pd.cut(values, bins=bins, labels=labels, right=False, include_lowest=True)
        return target

    def get_feature_matrix(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Build the feature matrix for model training/inference.
        """
        drop_cols = {
            "target_severity",
            "individuals_affected",
            "entity_name",
            "breach_date",
            "description",
            "state",
            "year_month",
        }

        cols_to_use: List[str] = []
        for col in features_df.columns:
            if col in drop_cols:
                continue
            if col.endswith("_raw"):
                continue
            cols_to_use.append(col)

        if not cols_to_use:
            raise ValueError("No feature columns remaining after filtering.")

        self.feature_columns = cols_to_use
        X = features_df[self.feature_columns].fillna(0)
        return X

    def train(self, features_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Train the severity prediction model and compute evaluation metadata.
        """
        # Features and target
        X = self.get_feature_matrix(features_df)
        y = self.prepare_target(features_df)

        # Remove rows with NaN targets
        mask = y.notna()
        X = X.loc[mask]
        y = y.loc[mask]

        if len(y.unique()) < 2:
            raise ValueError("Need at least two classes in target to train a classifier.")

        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y.astype(str))

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y_encoded,
            test_size=settings.test_size,
            stratify=y_encoded,
            random_state=settings.rf_random_state,
        )

        # Cross-validation
        skf = StratifiedKFold(
            n_splits=5,
            shuffle=True,
            random_state=settings.rf_random_state,
        )
        cv_scores = cross_val_score(
            self.model,
            X_train,
            y_train,
            cv=skf,
            scoring="f1_weighted",
            n_jobs=-1,
        )

        # Fit on training data
        self.model.fit(X_train, y_train)

        # Evaluate on test set
        y_pred = self.model.predict(X_test)

        report_dict = classification_report(
            y_test,
            y_pred,
            output_dict=True,
            zero_division=0,
        )
        cm = confusion_matrix(y_test, y_pred)

        logger.info("Severity model classification report:")
        logger.info(json.dumps(report_dict, indent=2))

        # Feature importances
        importances = self.model.feature_importances_
        feature_importance_df = pd.DataFrame(
            {
                "feature": self.feature_columns,
                "importance": importances,
            }
        ).sort_values("importance", ascending=False)

        # Class distribution
        class_counts = pd.Series(y_encoded).value_counts().to_dict()
        class_distribution = {
            self.label_encoder.inverse_transform([k])[0]: int(v)
            for k, v in class_counts.items()
        }

        metadata: Dict[str, Any] = {
            "model_type": "RandomForestClassifier",
            "n_estimators": self.model.n_estimators,
            "max_depth": self.model.max_depth,
            "n_features": int(X.shape[1]),
            "n_samples_train": int(X_train.shape[0]),
            "n_samples_test": int(X_test.shape[0]),
            "class_distribution": class_distribution,
            "cv_f1_mean": round(float(np.mean(cv_scores)), 4),
            "cv_f1_std": round(float(np.std(cv_scores)), 4),
            "test_classification_report": report_dict,
            "confusion_matrix": cm.tolist(),
            "top_15_features": feature_importance_df.head(15).to_dict(orient="records"),
            "trained_at": datetime.now(timezone.utc).isoformat(),
        }

        self.metadata = metadata
        return metadata

    def _ensure_feature_columns(self) -> None:
        if not self.feature_columns:
            raise ValueError("Model feature_columns are not set. Train or load the model first.")

    def _prepare_input_matrix(self, X: Any) -> pd.DataFrame:
        """
        Ensure input data has the correct feature columns, filling missing with 0.
        """
        self._ensure_feature_columns()

        if isinstance(X, pd.DataFrame):
            X_df = X.copy()
        else:
            # Assume array-like
            X_df = pd.DataFrame(X, columns=self.feature_columns)

        for col in self.feature_columns:
            if col not in X_df.columns:
                X_df[col] = 0

        # Reorder and fill NaNs
        X_df = X_df[self.feature_columns].fillna(0)
        return X_df

    def predict(self, X: Any) -> Dict[str, Any]:
        """
        Predict severity classes and probabilities for new data.
        """
        X_matrix = self._prepare_input_matrix(X)

        preds_encoded = self.model.predict(X_matrix)
        proba = self.model.predict_proba(X_matrix)
        classes = list(self.label_encoder.classes_)

        predictions = self.label_encoder.inverse_transform(preds_encoded).tolist()
        probabilities: Dict[str, List[float]] = {
            cls: proba[:, i].tolist() for i, cls in enumerate(classes)
        }
        confidence = proba.max(axis=1).tolist()

        return {
            "prediction": predictions,
            "probabilities": probabilities,
            "confidence": confidence,
        }

    def save(self, model_dir: Optional[Path | str] = None) -> None:
        """
        Save model, label encoder, feature columns, and metadata to disk.
        """
        target_dir = Path(model_dir) if model_dir is not None else Path(settings.model_dir)
        target_dir.mkdir(parents=True, exist_ok=True)

        model_path = target_dir / "severity_model.joblib"
        le_path = target_dir / "severity_label_encoder.joblib"
        features_path = target_dir / "severity_features.joblib"
        metadata_path = target_dir / "severity_metadata.json"

        joblib.dump(self.model, model_path)
        joblib.dump(self.label_encoder, le_path)
        joblib.dump(self.feature_columns, features_path)

        with metadata_path.open("w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=2, default=str)

        logger.info(f"Saved severity model artifacts to {target_dir}")

    def load(self, model_dir: Optional[Path | str] = None) -> None:
        """
        Load model, label encoder, feature columns, and metadata from disk.
        """
        target_dir = Path(model_dir) if model_dir is not None else Path(settings.model_dir)

        model_path = target_dir / "severity_model.joblib"
        le_path = target_dir / "severity_label_encoder.joblib"
        features_path = target_dir / "severity_features.joblib"
        metadata_path = target_dir / "severity_metadata.json"

        self.model = joblib.load(model_path)
        self.label_encoder = joblib.load(le_path)
        self.feature_columns = joblib.load(features_path)

        if metadata_path.exists():
            with metadata_path.open("r", encoding="utf-8") as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {}

        logger.info(f"Loaded severity model artifacts from {target_dir}")

