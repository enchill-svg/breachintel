from __future__ import annotations

import pandas as pd
import streamlit as st

from breachintel.utils.cache import load_data
from breachintel.ml.forecaster import BreachForecaster
from breachintel.visualization.charts import create_forecast_chart

from app.components.filters import (
    configure_time_filters,
    render_active_filter_bar,
    render_sidebar_filters,
)
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
        "Projected monthly breach volumes — use as directional guidance, not precise prediction."
    )

    df = get_data()

    # Sidebar: date range first, then breach/entity/geography (same as Home)
    st.sidebar.header("Filters")
    time_filtered_df, time_meta = configure_time_filters(df, in_sidebar=True)
    filtered_df, filter_state = render_sidebar_filters(time_filtered_df)

    try:
        forecaster = load_or_train_forecaster(filtered_df)
        # Use existing forecast if already trained; otherwise predict from fitted model (no fit)
        if forecaster._forecast_df is not None:
            forecast_df = forecaster._forecast_df
        else:
            # Loaded from disk: prepare history, extend future, predict (Prophet allows only one fit)
            forecaster.prepare_data(filtered_df)
            future = forecaster.model.make_future_dataframe(
                periods=forecaster.forecast_months,
                freq="ME",
            )
            forecast_df = forecaster.model.predict(future)
            forecaster._forecast_df = forecast_df
        summary = forecaster.get_forecast_summary()
    except Exception:
        st.warning(
            "The forecast model could not be loaded. Please run "
            "'python scripts/train_models.py' to train the models.",
            icon="⚠️",
        )
        render_footer()
        return

    history = forecaster._history_df.copy()  # type: ignore[attr-defined]
    history = history.rename(columns={"ds": "date", "y": "breach_count"})
    history["date"] = pd.to_datetime(history["date"], errors="coerce")

    forecast_chart_df = forecast_df.rename(
        columns={"ds": "date", "yhat": "forecast", "yhat_lower": "lower", "yhat_upper": "upper"}
    )
    forecast_chart_df["date"] = pd.to_datetime(forecast_chart_df["date"], errors="coerce")

    # Align with dashboard date filter: only show from start_date onward
    start_date = time_meta.get("start_date") if time_meta else None
    if start_date is not None:
        start_dt = pd.Timestamp(start_date)
        history = history[history["date"].dt.date >= start_date]
        forecast_chart_df = forecast_chart_df[forecast_chart_df["date"] >= start_dt]

    # Prevent negative confidence interval or forecast (clip to >= 0)
    forecast_chart_df = forecast_chart_df.copy()
    forecast_chart_df["lower"] = forecast_chart_df["lower"].clip(lower=0)
    forecast_chart_df["forecast"] = forecast_chart_df["forecast"].clip(lower=0)

    fig = create_forecast_chart(
        forecast_df=forecast_chart_df[["date", "forecast", "lower", "upper"]],
        actual_df=history[["date", "breach_count"]],
    )
    st.plotly_chart(fig, use_container_width=True)

    # Narrative under forecast chart
    future_mask = forecast_chart_df["date"] > history["date"].max()
    future = forecast_chart_df.loc[future_mask]
    if not future.empty and len(future) >= 6:
        yhat = future["forecast"]
        lo, hi = float(yhat.min()), float(yhat.max())
        start = future["date"].min().date()
        end = future["date"].max().date()
        direction = summary["trend_direction"]
        # Use HTML <strong> so bold renders inside the styled paragraph (markdown ** is not parsed in raw HTML)
        text = (
            f"Over the next 24 months (<strong>{start}</strong> – <strong>{end}</strong>), "
            f"predicted monthly breaches range from <strong>{lo:.1f}</strong> to <strong>{hi:.1f}</strong> "
            f"incidents, with an overall <strong>{direction}</strong> trend."
        )
    else:
        text = (
            "Forecast horizon is too short to summarize; try expanding the time range or ensuring "
            "sufficient historical data is available."
        )

    st.markdown(
        f"<p style='color:#9CA3AF;font-size:0.85rem;'>{text}</p>",
        unsafe_allow_html=True,
    )

    # Summary metrics
    st.subheader("Forecast Summary")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            label="Avg predicted monthly breaches",
            value=f"{summary['avg_predicted_monthly']:.1f}",
            help="Average of the model’s monthly forecasts over the next 24 months.",
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
            help="Whether predicted breach counts trend up or down across the forecast window.",
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

