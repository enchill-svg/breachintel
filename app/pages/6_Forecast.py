from __future__ import annotations

import pandas as pd
import streamlit as st

from breachintel.utils.cache import load_data
from breachintel.ml.forecaster import BreachForecaster
from breachintel.visualization.charts import create_forecast_chart

from app.components.filters import render_sidebar_filters
from app.components.footer import render_footer


st.set_page_config(
    page_title="BreachIntel — Breach Forecast",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data(ttl=3600)
def get_data() -> pd.DataFrame:
    return load_data()


@st.cache_resource
def load_or_train_forecaster(df: pd.DataFrame) -> BreachForecaster:
    forecaster = BreachForecaster()
    from pathlib import Path
    from breachintel.config import settings

    model_path = Path(settings.model_dir) / "breach_forecaster.json"
    try:
        if model_path.exists():
            forecaster.load()
        else:
            forecaster.train_and_forecast(df)
            forecaster.save()
    except Exception:
        # Fallback: train on the fly if loading fails
        forecaster.train_and_forecast(df)
        forecaster.save()
    return forecaster


def main() -> None:
    st.title("🔮 Breach Forecast")
    st.caption(
        "Forecast monthly healthcare breach volumes using a Prophet-based time-series model."
    )

    df = get_data()
    filtered_df = render_sidebar_filters(df)

    try:
        forecaster = load_or_train_forecaster(filtered_df)
        forecast_df = forecaster.train_and_forecast(filtered_df)
        summary = forecaster.get_forecast_summary()
    except Exception as exc:
        st.warning(
            "Unable to train or load the breach forecast model. "
            "Ensure the Prophet dependency is installed and data is available.",
            icon="⚠️",
        )
        st.exception(exc)
        render_footer()
        return

    # Forecast chart
    st.subheader("24‑Month Breach Forecast")

    history = forecaster._history_df.copy()  # type: ignore[attr-defined]
    history = history.rename(columns={"ds": "date", "y": "breach_count"})

    forecast_chart_df = forecast_df.rename(
        columns={"ds": "date", "yhat": "forecast", "yhat_lower": "lower", "yhat_upper": "upper"}
    )

    fig = create_forecast_chart(
        forecast_df=forecast_chart_df[["date", "forecast", "lower", "upper"]],
        actual_df=history[["date", "breach_count"]],
    )
    st.plotly_chart(fig, use_container_width=True)

    # Summary metrics
    st.subheader("Forecast Summary")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            label="Avg predicted monthly breaches",
            value=f"{summary['avg_predicted_monthly']:.1f}",
        )
    with col2:
        st.metric(
            label="Predicted range (monthly)",
            value=f"{summary['min_predicted_monthly']:.1f} – {summary['max_predicted_monthly']:.1f}",
        )
    with col3:
        st.metric(
            label="Trend direction",
            value=summary["trend_direction"].title(),
        )

    # Interpretation / caveats
    st.markdown("---")
    st.subheader("How to Interpret This Forecast")
    st.markdown(
        """
        The forecast projects likely monthly breach volumes over the next 24 months based on
        historical incident counts. Higher predicted values and an *increasing* trend suggest
        intensifying breach activity and a need for stronger controls, while a *decreasing* trend
        may reflect improvements in security posture, regulatory enforcement, or reporting.
        """
    )

    st.markdown("#### Important limitations")
    st.markdown(
        """
        - The model assumes that historical patterns continue; abrupt regulatory changes or
          new attack techniques may invalidate these assumptions.
        - Forecasts are based on reported breaches affecting 500+ individuals and may not
          capture smaller or unreported incidents.
        - Seasonality and structural breaks are approximated; confidence intervals reflect
          statistical uncertainty but not all real-world risks.
        """
    )

    render_footer()


if __name__ == "__main__":
    main()

