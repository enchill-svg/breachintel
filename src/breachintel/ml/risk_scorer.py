from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


@dataclass
class RiskScorer:
    """
    Organization-level breach risk scoring.
    """

    df: pd.DataFrame
    severity_model: Optional[Any] = None
    entity_risk: Dict[str, float] = field(default_factory=dict, init=False)
    state_risk: Dict[str, float] = field(default_factory=dict, init=False)
    trend_multiplier: float = field(default=1.0, init=False)

    def __post_init__(self) -> None:
        self._compute_baselines()

    def _compute_baselines(self) -> None:
        """
        Compute baseline risk statistics from historical data.
        """
        df = self.df.copy()

        # Entity-level baseline: normalized frequency by entity_type
        if "entity_type" in df.columns:
            counts = df["entity_type"].value_counts(normalize=True)
            self.entity_risk = counts.to_dict()
        else:
            self.entity_risk = {}

        # State-level baseline: frequency normalized by max state count
        if "state" in df.columns:
            state_counts = df["state"].value_counts()
            max_count = state_counts.max() if not state_counts.empty else 0
            if max_count > 0:
                self.state_risk = (state_counts / max_count).to_dict()
            else:
                self.state_risk = {}
        else:
            self.state_risk = {}

        # Trend multiplier: latest year vs previous year breach counts
        if "breach_date" in df.columns:
            df = df.copy()
            df["breach_date"] = pd.to_datetime(df["breach_date"], errors="coerce")
            df = df.dropna(subset=["breach_date"])
            if not df.empty:
                df["year"] = df["breach_date"].dt.year
                yearly_counts = df["year"].value_counts().sort_index()
                if len(yearly_counts) >= 2:
                    latest_year = yearly_counts.index.max()
                    prev_year = latest_year - 1
                    latest_count = int(yearly_counts.get(latest_year, 0))
                    prev_count = int(yearly_counts.get(prev_year, 0))
                    if prev_count > 0:
                        self.trend_multiplier = latest_count / prev_count
                    else:
                        self.trend_multiplier = 1.0
                else:
                    self.trend_multiplier = 1.0
            else:
                self.trend_multiplier = 1.0
        else:
            self.trend_multiplier = 1.0

    def _score_entity(self, entity_type: Optional[str]) -> float:
        if not self.entity_risk:
            return 50.0

        if entity_type is None:
            base = float(np.mean(list(self.entity_risk.values())))
        else:
            base = float(self.entity_risk.get(entity_type, np.mean(list(self.entity_risk.values()))))

        return float(np.clip(base * 100.0, 0.0, 100.0))

    def _score_state(self, state: Optional[str]) -> float:
        if not self.state_risk:
            return 50.0

        if state is None:
            base = float(np.mean(list(self.state_risk.values())))
        else:
            base = float(self.state_risk.get(state, np.mean(list(self.state_risk.values()))))

        return float(np.clip(base * 100.0, 0.0, 100.0))

    def _score_trend(self) -> float:
        """
        Map the trend multiplier into a 0–100 score.
        """
        m = self.trend_multiplier
        if not np.isfinite(m) or m <= 0:
            return 50.0

        # Cap multiplier at 2x for scoring to avoid extreme values
        capped = min(m, 2.0)
        score = 50.0 + (capped - 1.0) * 50.0  # 1.0 -> 50, 2.0 -> 100
        return float(np.clip(score, 0.0, 100.0))

    def _risk_level(self, score: float) -> str:
        if score < 26:
            return "Low"
        if score < 51:
            return "Moderate"
        if score < 76:
            return "High"
        return "Critical"

    def score(
        self,
        entity_type: Optional[str],
        state: Optional[str],
        breach_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Compute composite risk score for a prospective organization context.
        """
        entity_score = self._score_entity(entity_type)
        state_score = self._score_state(state)
        trend_score = self._score_trend()

        weights = {
            "entity": 0.30,
            "state": 0.30,
            "trend": 0.40,
        }

        overall = (
            entity_score * weights["entity"]
            + state_score * weights["state"]
            + trend_score * weights["trend"]
        )
        overall = float(np.clip(overall, 0.0, 100.0))

        risk_level = self._risk_level(overall)

        interpretation = (
            f"Overall risk is {risk_level} ({overall:.1f}/100), combining "
            f"entity-type exposure ({entity_score:.1f}), state exposure ({state_score:.1f}), "
            f"and macro breach trend ({trend_score:.1f})."
        )

        return {
            "overall_score": overall,
            "risk_level": risk_level,
            "components": {
                "entity_score": entity_score,
                "state_score": state_score,
                "trend_score": trend_score,
            },
            "weights": weights,
            "trend_multiplier": float(self.trend_multiplier),
            "breach_type": breach_type,
            "interpretation": interpretation,
        }

