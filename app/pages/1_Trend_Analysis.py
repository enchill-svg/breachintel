from __future__ import annotations

from typing import Dict

import pandas as pd
import streamlit as st

from breachintel.utils.cache import load_data
from breachintel.analysis.trends import TrendAnalyzer
from breachintel.visualization.charts import (
    create_overview_chart,
    create_yoy_growth,
    create_breach_type_area,
)

from app.components.filters import render_sidebar_filters
from app.components.footer import render_footer


st.set_page_config(
    page_title="BreachIntel — Trend Analysis",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data(ttl=3600)
def get_data() -> pd.DataFrame:
    return load_data()


def compute_trend_data(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    analyzer = TrendAnalyzer()
    monthly = analyzer.compute_monthly_trends(df)
    yearly = analyzer.compute_yearly_trends(df)
    return {"monthly": monthly, "yearly": yearly}


def main() -> None:
    st.title("📈 Trend Analysis")
    st.caption(
        "Explore how healthcare data breaches evolve over time, from high-level "
        "monthly patterns to year-over-year shifts in incident volume."
    )

    df = get_data()
    filtered_df = render_sidebar_filters(df)

    trend_data = compute_trend_data(filtered_df)
    monthly = trend_data["monthly"]
    yearly = trend_data["yearly"]

    # Monthly vs Yearly tabs
    tab_monthly, tab_yearly = st.tabs(["Monthly Trends", "Yearly Trends"])

    with tab_monthly:
        st.subheader("Monthly Breach Activity")
        monthly_chart_df = monthly.reset_index().rename(columns={"breach_date": "date"})
        fig_overview = create_overview_chart(monthly_chart_df)
        st.plotly_chart(fig_overview, use_container_width=True)

        # Inflection points
        analyzer = TrendAnalyzer()
        inflections = analyzer.detect_inflection_points(monthly)
        if inflections:
            for point in inflections:
                dt = point["date"]
                ma_val = point["breach_count_ma"]
                direction = point["direction"]
                direction_label = "upward" if direction == "up" else "downward"
                st.info(
                    f"Trend inflection on **{dt.date()}**: 12-month average "
                    f"at **{ma_val:.1f}** breaches, turning **{direction_label}**.",
                    icon="🔍",
                )
        else:
            st.info(
                "No clear inflection points detected in the 12‑month moving average.",
                icon="ℹ️",
            )

    with tab_yearly:
        st.subheader("Yearly Breach Metrics")

        display_cols = [
            "breach_count",
            "total_affected",
            "avg_affected",
            "median_affected",
            "unique_entities",
            "unique_states",
            "yoy_count_change",
            "yoy_affected_change",
        ]
        yearly_display = yearly[display_cols].copy()
        yearly_display.index.name = "year"

        st.dataframe(
            yearly_display.style.format(
                {
                    "breach_count": "{:,.0f}",
                    "total_affected": "{:,.0f}",
                    "avg_affected": "{:,.1f}",
                    "median_affected": "{:,.1f}",
                    "yoy_count_change": "{:+.1f}%",
                    "yoy_affected_change": "{:+.1f}%",
                }
            ),
            use_container_width=True,
        )

        yoy_df = yearly.reset_index()[["year", "yoy_count_change"]].dropna()
        fig_yoy = create_yoy_growth(yoy_df)
        st.plotly_chart(fig_yoy, use_container_width=True)

    st.markdown("---")
    st.subheader("Evolution of Breach Types")

    # Prepare data for stacked area chart
    breach_type_df = filtered_df.copy()
    breach_type_df["date"] = breach_type_df["breach_date"]
    fig_breach_type = create_breach_type_area(breach_type_df)
    st.plotly_chart(fig_breach_type, use_container_width=True)

    render_footer()


if __name__ == "__main__":
    main()

