from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

from breachintel.utils.cache import load_data
from breachintel.ml.severity_model import SeverityModel
from breachintel.ml.risk_scorer import RiskScorer
from breachintel.ml.explainer import SeverityExplainer

from app.components.filters import render_sidebar_filters
from app.components.footer import render_footer
from app.components.metrics import render_severity_badge


st.set_page_config(
    page_title="BreachIntel — Severity Predictor",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data(ttl=3600)
def get_data() -> pd.DataFrame:
    return load_data()


@st.cache_resource
def load_severity_model() -> Optional[SeverityModel]:
    try:
        model = SeverityModel()
        model.load()
        return model
    except Exception:
        return None


def build_feature_vector(
    feature_columns: List[str],
    entity_type: str,
    breach_type: str,
    breach_location: str,
    state: str,
    year: int,
    business_associate: bool,
) -> Dict[str, Any]:
    """
    Build a one-hot style feature vector aligned with the training feature columns.
    """
    features: Dict[str, Any] = {col: 0 for col in feature_columns}

    # Simple conventions that match likely one-hot encoding from training
    def set_if_present(col_name: str) -> None:
        if col_name in features:
            features[col_name] = 1

    set_if_present(f"entity_type_{entity_type}")
    set_if_present(f"breach_type_{breach_type}")
    set_if_present(f"breach_location_{breach_location}")
    set_if_present(f"state_{state}")

    # Year-related features
    set_if_present(f"year_{year}")
    if "year" in features:
        features["year"] = year

    # Business associate flag
    set_if_present("is_business_associate")
    if "business_associate" in features:
        features["business_associate"] = int(business_associate)

    return features


def main() -> None:
    st.title("🎯 Breach Severity Predictor")
    st.caption(
        "Interactively estimate the likely severity of a prospective healthcare data "
        "breach and understand which factors drive the model's prediction."
    )

    df = get_data()
    # Use filters only for deriving available choices; prediction itself is independent
    filtered_df = render_sidebar_filters(df)

    model = load_severity_model()
    if model is None or model.feature_columns is None:
        st.warning(
            "Severity prediction model artifacts are not available. "
            "Please run `python scripts/train_models.py` to train and save the models "
            "before using this page.",
            icon="⚠️",
        )
        render_footer()
        return

    # Input form
    st.subheader("Configure a Hypothetical Breach Scenario")

    entity_options = ["Healthcare Provider", "Health Plan", "Business Associate", "Healthcare Clearing House"]
    breach_type_options = [
        "Hacking/IT Incident",
        "Unauthorized Access/Disclosure",
        "Theft",
        "Loss",
        "Improper Disposal",
        "Other",
    ]
    breach_location_options = [
        "Network Server",
        "Email",
        "Paper/Films",
        "Portable Device",
        "Desktop",
        "EMR",
        "Other",
    ]

    states = sorted(filtered_df["state"].dropna().unique().tolist()) if "state" in filtered_df.columns else []

    with st.form("severity_predictor_form"):
        col_left, col_right = st.columns(2)

        with col_left:
            entity_type = st.selectbox("Entity type", options=entity_options, index=0)
            breach_type = st.selectbox("Breach type", options=breach_type_options, index=0)
            breach_location = st.selectbox(
                "Breach location",
                options=breach_location_options,
                index=0,
            )

        with col_right:
            state = st.selectbox("State", options=states or ["CA"])
            year = st.slider("Year", min_value=2020, max_value=2026, value=2025)
            business_associate = st.checkbox("Business associate involved?", value=False)

        submitted = st.form_submit_button("🔮 Predict Severity", type="primary", use_container_width=True)

    if submitted:
        feature_vector = build_feature_vector(
            model.feature_columns,
            entity_type=entity_type,
            breach_type=breach_type,
            breach_location=breach_location,
            state=state,
            year=year,
            business_associate=business_associate,
        )
        X_df = pd.DataFrame([feature_vector])

        # Predictions
        result = model.predict(X_df)
        prediction = result["prediction"][0]
        confidence = float(result["confidence"][0]) if result["confidence"] else 0.0
        confidence_pct = confidence * 100.0

        # Risk score
        scorer = RiskScorer(df)
        risk = scorer.score(entity_type=entity_type, state=state, breach_type=breach_type)

        # Layout for results
        st.subheader("Prediction Results")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("###### Predicted Severity")
            st.markdown(render_severity_badge(prediction), unsafe_allow_html=True)

        with col2:
            st.markdown("###### Model Confidence")
            st.metric(label="Confidence", value=f"{confidence_pct:.1f}%")

        with col3:
            st.markdown("###### Risk Score")
            st.metric(
                label="Composite Risk",
                value=f"{risk['overall_score']:.1f}/100",
                help=risk.get("interpretation", ""),
            )

        # SHAP explanation
        st.markdown("---")
        st.subheader("Why did the model predict this severity?")

        try:
            explainer = SeverityExplainer(model=model.model, feature_columns=model.feature_columns)
            shap_info = explainer.explain_single(X_df.iloc[0])
            top_features = shap_info["top_contributing_features"]
            if top_features:
                top_features = sorted(
                    top_features,
                    key=lambda x: abs(x["shap_value"]),
                    reverse=True,
                )[:5]

                expl_cols = st.columns(len(top_features))
                for col, feat in zip(expl_cols, top_features):
                    with col:
                        st.metric(
                            label=feat["feature"],
                            value=f"{feat['shap_value']:+.3f}",
                        )
            else:
                st.info(
                    "SHAP could not identify top contributing features for this prediction.",
                    icon="ℹ️",
                )
        except Exception:
            st.warning(
                "SHAP explanation is unavailable. Ensure SHAP is installed and model artifacts "
                "were trained with compatible settings.",
                icon="⚠️",
            )

    render_footer()


if __name__ == "__main__":
    main()

