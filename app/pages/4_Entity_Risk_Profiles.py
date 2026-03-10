from __future__ import annotations

import pandas as pd
import streamlit as st

from breachintel.utils.cache import load_data
from breachintel.analysis.entity_profiling import EntityProfiler
from breachintel.visualization.charts import create_entity_comparison, create_breach_type_area

from app.components.filters import render_sidebar_filters
from app.components.footer import render_footer


st.set_page_config(
    page_title="BreachIntel — Entity Risk Profiles",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data(ttl=3600)
def get_data() -> pd.DataFrame:
    return load_data()


def main() -> None:
    st.title("🏥 Entity Risk Profiles")
    st.caption(
        "Compare breach exposure across healthcare entity types and identify repeat "
        "offenders driving systemic risk."
    )

    df = get_data()
    filtered_df = render_sidebar_filters(df)

    profiler = EntityProfiler()
    entity_summary = profiler.compute_entity_summary(filtered_df)
    entity_trend = profiler.compute_entity_trend(filtered_df)
    repeat_offenders = profiler.find_most_breached_entities(filtered_df, top_n=20)

    # Entity type comparison bar chart
    st.subheader("Entity Type Comparison")
    comp_df = (
        entity_summary.reset_index()
        .rename(columns={"entity_type": "entity_type", "breach_count": "breach_count"})
    )
    fig_entities = create_entity_comparison(comp_df[["entity_type", "breach_count"]])
    st.plotly_chart(fig_entities, use_container_width=True)

    # Entity summary table
    st.subheader("Entity Summary")
    display_cols = [
        "breach_count",
        "total_affected",
        "avg_affected",
        "median_affected",
        "max_affected",
        "unique_entities",
        "pct_of_breaches",
        "repeat_offenders",
    ]
    summary_display = entity_summary[display_cols].copy()
    summary_display.index.name = "entity_type"
    st.dataframe(
        summary_display.style.format(
            {
                "breach_count": "{:,.0f}",
                "total_affected": "{:,.0f}",
                "avg_affected": "{:,.1f}",
                "median_affected": "{:,.1f}",
                "max_affected": "{:,.0f}",
                "pct_of_breaches": "{:.1f}%",
            }
        ),
        use_container_width=True,
    )

    st.markdown("---")

    # Repeat offenders section
    st.subheader("Repeat Offenders")
    repeat_display = repeat_offenders.reset_index()
    repeat_display = repeat_display[
        ["entity_name", "breach_count", "total_affected", "entity_type", "state"]
    ]
    st.dataframe(
        repeat_display.style.format(
            {
                "breach_count": "{:,.0f}",
                "total_affected": "{:,.0f}",
            }
        ),
        use_container_width=True,
    )

    st.markdown("---")

    # Entity type trend over time
    st.subheader("Entity Type Trend Over Time")
    trend_df = entity_trend.copy()
    trend_df["date"] = pd.to_datetime(trend_df["year"].astype(str) + "-01-01")
    fig_trend = create_breach_type_area(
        trend_df.rename(columns={"entity_type": "breach_type"})
    )
    st.plotly_chart(fig_trend, use_container_width=True)

    render_footer()


if __name__ == "__main__":
    main()

