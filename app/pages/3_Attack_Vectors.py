from __future__ import annotations

import pandas as pd
import streamlit as st

from breachintel.utils.cache import load_data
from breachintel.analysis.attack_vectors import AttackVectorAnalyzer
from breachintel.visualization.charts import create_breach_type_area, create_entity_comparison

from app.components.filters import render_sidebar_filters
from app.components.footer import render_footer
from app.components.metrics import render_severity_badge


st.set_page_config(
    page_title="BreachIntel — Attack Vector Analysis",
    page_icon="⚔️",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data(ttl=3600)
def get_data() -> pd.DataFrame:
    return load_data()


def main() -> None:
    st.title("⚔️ Attack Vector Analysis")

    df = get_data()
    filtered_df = render_sidebar_filters(df)

    analyzer = AttackVectorAnalyzer()
    vector_summary = analyzer.compute_vector_summary(filtered_df)
    evolution_df = analyzer.compute_vector_evolution(filtered_df)
    location_df = analyzer.compute_location_analysis(filtered_df)
    severity_matrix = analyzer.compute_vector_severity_matrix(filtered_df)

    # Key insight callout
    total_pct = vector_summary["percentage"].sum()
    hacking_pct = (
        vector_summary.loc["Hacking/IT Incident", "percentage"]
        if "Hacking/IT Incident" in vector_summary.index
        else None
    )
    if hacking_pct is not None:
        st.info(
            f"Hacking/IT Incidents now account for **{hacking_pct:.1f}%** "
            f"of all healthcare breaches.",
            icon="💡",
        )
    else:
        st.info(
            "Hacking/IT Incident records are not present in the current filtered view.",
            icon="ℹ️",
        )

    # Breach type summary table
    st.subheader("Breach Type Summary")
    summary_display = vector_summary.copy()
    summary_display["percentage"] = summary_display["percentage"].round(1)
    st.dataframe(
        summary_display.style.format(
            {
                "count": "{:,.0f}",
                "total_affected": "{:,.0f}",
                "avg_affected": "{:,.1f}",
                "median_affected": "{:,.1f}",
                "max_affected": "{:,.0f}",
                "percentage": "{:.1f}%",
            }
        ),
        use_container_width=True,
    )

    st.markdown("---")

    # Breach type evolution stacked area chart
    st.subheader("Evolution of Breach Types Over Time")
    bt_df = evolution_df.copy()
    # create_breach_type_area expects columns: year, breach_type, date (or we adapt)
    # We will convert year to a datetime year-start for the x-axis and call it 'date'.
    bt_df["date"] = pd.to_datetime(bt_df["year"].astype(str) + "-01-01")
    fig_bt = create_breach_type_area(
        bt_df.rename(columns={"breach_type": "breach_type"})
    )
    st.plotly_chart(fig_bt, use_container_width=True)

    st.markdown("---")

    # Breach location analysis - horizontal bar chart
    st.subheader("Where Are Breaches Occurring?")
    location_summary = (
        location_df.reset_index()
        .rename(columns={"breach_location": "entity_type", "count": "breach_count"})
    )
    fig_location = create_entity_comparison(location_summary)
    st.plotly_chart(fig_location, use_container_width=True)

    # Severity matrix as styled table
    st.subheader("Severity Matrix (Breach Type × Severity)")
    severity_display = severity_matrix.copy()
    st.dataframe(
        severity_display.style.format("{:,.0f}"),
        use_container_width=True,
    )

    render_footer()


if __name__ == "__main__":
    main()

