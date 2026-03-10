from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import shap


@dataclass
class SeverityExplainer:
    """
    SHAP-based explainer for severity classification models.
    """

    model: Any
    feature_columns: List[str]

    def __post_init__(self) -> None:
        self.explainer = shap.TreeExplainer(self.model)

    def _to_feature_frame(self, X: Any) -> pd.DataFrame:
        """
        Normalize input to a DataFrame with the correct feature columns.
        """
        if isinstance(X, pd.Series):
            df = X.to_frame().T
        elif isinstance(X, pd.DataFrame):
            df = X.copy()
        else:
            df = pd.DataFrame([X], columns=self.feature_columns)

        for col in self.feature_columns:
            if col not in df.columns:
                df[col] = 0

        df = df[self.feature_columns]
        return df

    def explain_single(self, X_single: Any) -> Dict[str, Any]:
        """
        Explain a single prediction.
        """
        X_df = self._to_feature_frame(X_single)

        # SHAP values shape: (n_classes, n_samples, n_features) for multi-class
        shap_values = self.explainer.shap_values(X_df)

        # Determine predicted class index
        pred = self.model.predict(X_df)[0]
        classes = getattr(self.model, "classes_", None)
        if classes is not None:
            try:
                class_index = int(np.where(classes == pred)[0][0])
            except IndexError:
                class_index = 0
        else:
            class_index = 0

        if isinstance(shap_values, list):
            sv = np.array(shap_values[class_index][0])
        else:
            sv = np.array(shap_values[0])

        features = self.feature_columns

        # Sort by absolute SHAP magnitude
        abs_sv = np.abs(sv)
        order = np.argsort(-abs_sv)

        # Top contributing (positive)
        pos_idx = [i for i in order if sv[i] > 0][:5]
        top_contributing = [
            {
                "feature": features[i],
                "shap_value": float(sv[i]),
                "abs_shap": float(abs_sv[i]),
            }
            for i in pos_idx
        ]

        # Top opposing (negative)
        neg_idx = [i for i in np.argsort(sv) if sv[i] < 0][:5]
        top_opposing = [
            {
                "feature": features[i],
                "shap_value": float(sv[i]),
                "abs_shap": float(abs_sv[i]),
            }
            for i in neg_idx
        ]

        return {
            "shap_values": sv.tolist(),
            "features": features,
            "top_contributing_features": top_contributing,
            "top_opposing_features": top_opposing,
        }

    def compute_global_importance(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Compute global feature importance using mean absolute SHAP values.
        """
        if len(X) == 0:
            return pd.DataFrame(columns=["feature", "mean_abs_shap"])

        # Sample for performance
        if len(X) > 1000:
            X_sample = X.sample(n=1000, random_state=42)
        else:
            X_sample = X

        shap_values = self.explainer.shap_values(X_sample)

        if isinstance(shap_values, list):
            # Multi-class: take mean abs across samples per class, then average across classes
            class_means = [np.abs(sv).mean(axis=0) for sv in shap_values]
            mean_abs_shap = np.mean(class_means, axis=0)
        else:
            mean_abs_shap = np.abs(shap_values).mean(axis=0)

        importance_df = pd.DataFrame(
            {
                "feature": self.feature_columns,
                "mean_abs_shap": mean_abs_shap,
            }
        ).sort_values("mean_abs_shap", ascending=False)

        return importance_df

