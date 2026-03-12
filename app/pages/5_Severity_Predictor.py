from __future__ import annotations

from typing import Any, Dict, List, Optional
from pathlib import Path
import sys

import pandas as pd
import plotly.express as px
import streamlit as st

from breachintel.utils.cache import load_data
from breachintel.ml.severity_model import SeverityModel
from breachintel.ml.risk_scorer import RiskScorer
from breachintel.ml.explainer import SeverityExplainer

# Ensure the project root (which contains the `app` package) is on sys.path so
# imports like `from app.components ...` work when the working directory is not
# the repository root (e.g., Streamlit Cloud).
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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

    # Business associate flag – align with training feature name(s)
    # Newer feature sets use 'has_business_associate'; older ones may use
    # 'is_business_associate' or a numeric 'business_associate' column.
    if "has_business_associate" in features:
        features["has_business_associate"] = int(business_associate)
    else:
        set_if_present("is_business_associate")
        if "business_associate" in features:
            features["business_associate"] = int(business_associate)

    return features


def main() -> None:
    st.title("🎯 Breach Severity Predictor")

    df = get_data()
    # Use filters only for deriving available choices; prediction itself is independent

    # Sidebar: date range first, then breach/entity/geography (same as Home)
    st.sidebar.header("Filters")
    time_filtered_df, time_meta = configure_time_filters(df, in_sidebar=True)
    # Sidebar filters (breach type, entity type, geography)
    filtered_df, filter_state = render_sidebar_filters(time_filtered_df)

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
            # Use a dropdown for year so users can click instead of dragging a slider
            year_options = list(range(2020, 2027))
            year = st.selectbox("Year", options=year_options, index=len(year_options) - 1)
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
            # Display confidence as a whole-number percentage (e.g. 68%)
            st.metric(label="Confidence", value=f"{confidence_pct:.0f}%")

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
            raw_feat_df = pd.DataFrame(
                {"feature": model.feature_columns, "importance": importances}
            ).sort_values("importance", ascending=False)

            # Clean up raw feature names (one‑hot encoded columns) for display
            def _humanize_feature(name: str) -> str:
                if name == "individuals_affected":
                    return "People Affected"

                mappings = {
                    "breach_type_": "Type: ",
                    "entity_type_": "Entity: ",
                    "breach_location_": "Location: ",
                    "location_": "Location: ",
                    "state_": "State: ",
                }
                for prefix, label in mappings.items():
                    if name.startswith(prefix):
                        tail = name[len(prefix) :].replace("_", " ")
                        return f"{label}{tail}"

                if name == "is_business_associate":
                    return "Business Associate Flag"
                if name == "business_associate":
                    return "Business Associate"

                # Fallback: replace underscores and title‑case
                return name.replace("_", " ").title()

            raw_feat_df["feature_label"] = raw_feat_df["feature"].apply(_humanize_feature)

            # Top 8 features for the chart
            feat_df = raw_feat_df.head(8).sort_values("importance", ascending=True)

            # Horizontal bar chart of feature importance with cleaned labels
            fig_importance = px.bar(
                feat_df,
                x="importance",
                y="feature_label",
                orientation="h",
                title=None,
            )
            fig_importance.update_layout(
                xaxis_title="Relative Importance",
                yaxis_title="Model Feature",
            )
            st.plotly_chart(fig_importance, use_container_width=True)

        else:
            st.info(
                "Feature importances are not available for this model type.",
                icon="ℹ️",
            )

    render_footer()


if __name__ == "__main__":
    main()

