from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

from breachintel.utils.cache import load_data
from breachintel.ml.severity_model import SeverityModel
from breachintel.ml.risk_scorer import RiskScorer
from breachintel.ml.explainer import SeverityExplainer

from app.components.filters import (
    configure_time_filters,
    render_active_filter_bar,
    render_sidebar_filters,
)
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

    # Top-of-page time controls
    time_filtered_df, time_meta = configure_time_filters(df)

    # Sidebar filters (breach type, entity type, geography)
    filtered_df, filter_state = render_sidebar_filters(time_filtered_df)

    # Contextual time summary + active filter chips
    if time_meta:
        start = time_meta.get("start_date")
        end = time_meta.get("end_date")
        total_incidents = time_meta.get("total_incidents", len(time_filtered_df))
        st.caption(
            f"Showing breaches from **{start}** to **{end}** "
            f"({int(total_incidents):,} incidents before additional filters)."
        )

    render_active_filter_bar(time_meta, filter_state)

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

        # Model explanation (feature importance proxy)
        st.markdown("---")
        st.subheader("Why did the model predict this severity?")

        importances = getattr(model.model, "feature_importances_", None)
        if importances is not None and model.feature_columns:
            feat_df = (
                pd.DataFrame(
                    {"feature": model.feature_columns, "importance": importances}
                )
                .sort_values("importance", ascending=False)
                .head(5)
            )
            cols = st.columns(len(feat_df))
            for col, (_, row_feat) in zip(cols, feat_df.iterrows()):
                with col:
                    st.metric(
                        label=row_feat["feature"],
                        value=f"{row_feat['importance']:.3f}",
                    )
        else:
            st.info(
                "Feature importances are not available for this model type.",
                icon="ℹ️",
            )

    render_footer()


if __name__ == "__main__":
    main()

