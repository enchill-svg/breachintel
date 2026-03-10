from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

from ..config import settings
from ..utils.logger import logger


@dataclass
class NLPAttackClassifier:
    """
    NLP attack type classifier using TF-IDF + Random Forest.
    """

    # Category definitions for weak supervision
    CATEGORY_KEYWORDS: Dict[str, List[str]] = field(
        default_factory=lambda: {
            "Ransomware": [
                "ransomware",
                "ransom",
                "encrypted",
                "encryption",
                "decrypt",
                "locked files",
                "crypto",
            ],
            "Phishing": [
                "phishing",
                "phish",
                "spear-phishing",
                "email compromise",
                "business email",
                "bec",
                "social engineering",
                "spoofed",
            ],
            "Unauthorized Access": [
                "unauthorized access",
                "former employee",
                "insider",
                "terminated",
                "snooping",
                "curiosity",
                "impermissible",
            ],
            "Physical Theft": [
                "stolen",
                "theft",
                "burglary",
                "break-in",
                "car",
                "vehicle",
                "missing laptop",
                "lost device",
            ],
            "Network Intrusion": [
                "hacking",
                "hacked",
                "vulnerability",
                "exploit",
                "sql injection",
                "brute force",
                "compromised server",
                "malware",
                "trojan",
            ],
            "Misconfiguration": [
                "misconfigured",
                "publicly accessible",
                "exposed",
                "unsecured",
                "open server",
                "s3 bucket",
                "cloud",
                "mailing error",
                "wrong recipient",
            ],
        }
    )

    pipeline: Pipeline = field(init=False)

    def __post_init__(self) -> None:
        self.pipeline = Pipeline(
            steps=[
                (
                    "tfidf",
                    TfidfVectorizer(
                        max_features=1000,
                        stop_words="english",
                        ngram_range=(1, 2),
                        min_df=3,
                        max_df=0.95,
                    ),
                ),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=150,
                        class_weight="balanced",
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        )

    def create_labels(self, descriptions: Iterable[str]) -> pd.Series:
        """
        Semi-supervised label creation based on keyword matching.
        """
        labels: List[str] = []
        for desc in descriptions:
            if not isinstance(desc, str):
                labels.append("Other")
                continue

            text = desc.lower()
            assigned = "Other"
            for category, keywords in self.CATEGORY_KEYWORDS.items():
                if any(keyword in text for keyword in keywords):
                    assigned = category
                    break
            labels.append(assigned)

        return pd.Series(labels, dtype="string")

    def train(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Train the NLP attack classifier on breach descriptions.
        """
        if "description" not in df.columns:
            raise KeyError("Input DataFrame must contain a 'description' column.")

        # Filter to rows with non-null, sufficiently long descriptions
        mask = df["description"].astype(str).str.len() > 20
        df_valid = df.loc[mask].copy()

        if len(df_valid) < 100:
            logger.warning(
                "Not enough valid descriptions to train NLPAttackClassifier "
                f"(found {len(df_valid)}, need at least 100)."
            )
            return {
                "status": "error",
                "message": "Not enough valid descriptions to train NLP classifier.",
                "n_valid": int(len(df_valid)),
            }

        X_text = df_valid["description"].astype(str)
        y = self.create_labels(X_text)

        label_counts = y.value_counts().to_dict()
        logger.info(f"NLPAttackClassifier label distribution: {label_counts}")

        X_train, X_test, y_train, y_test = train_test_split(
            X_text,
            y,
            test_size=0.2,
            stratify=y,
            random_state=settings.rf_random_state,
        )

        self.pipeline.fit(X_train, y_train)
        y_pred = self.pipeline.predict(X_test)

        report = classification_report(
            y_test,
            y_pred,
            output_dict=True,
            zero_division=0,
        )

        logger.info("NLPAttackClassifier classification report:")
        # Only log macro metrics at INFO level to avoid overly verbose logs
        macro_f1 = report.get("macro avg", {}).get("f1-score", np.nan)
        logger.info(f"Macro F1: {macro_f1:.4f} | support={int(report.get('accuracy', 0) * len(y_test))}")

        return {
            "status": "ok",
            "label_distribution": label_counts,
            "classification_report": report,
            "n_train": int(len(X_train)),
            "n_test": int(len(X_test)),
        }

    def predict(self, descriptions: Iterable[str]) -> pd.DataFrame:
        """
        Predict attack categories for new descriptions.
        """
        # Normalize to list of strings
        texts: List[str] = []
        for d in descriptions:
            texts.append("" if d is None or (isinstance(d, float) and np.isnan(d)) else str(d))

        if not texts:
            return pd.DataFrame(columns=["predicted_category", "confidence"])

        proba = self.pipeline.predict_proba(texts)
        preds = self.pipeline.classes_[np.argmax(proba, axis=1)]
        confidence = proba.max(axis=1)

        return pd.DataFrame(
            {
                "predicted_category": preds,
                "confidence": confidence,
            }
        )

    def save(self, model_dir: Optional[Path | str] = None) -> None:
        """
        Save the NLP pipeline to disk.
        """
        target_dir = Path(model_dir) if model_dir is not None else Path(settings.model_dir)
        target_dir.mkdir(parents=True, exist_ok=True)

        model_path = target_dir / "nlp_attack_classifier.joblib"
        joblib.dump(self.pipeline, model_path)

        logger.info(f"Saved NLPAttackClassifier pipeline to {model_path}")

    def load(self, model_dir: Optional[Path | str] = None) -> None:
        """
        Load the NLP pipeline from disk.
        """
        target_dir = Path(model_dir) if model_dir is not None else Path(settings.model_dir)
        model_path = target_dir / "nlp_attack_classifier.joblib"

        self.pipeline = joblib.load(model_path)
        logger.info(f"Loaded NLPAttackClassifier pipeline from {model_path}")

