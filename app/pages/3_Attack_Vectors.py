from __future__ import annotations

import pandas as pd
import streamlit as st

from breachintel.utils.cache import load_data
from breachintel.analysis.attack_vectors import AttackVectorAnalyzer
from breachintel.visualization.charts import create_breach_type_area, create_entity_comparison

from app.components.filters import (
    configure_time_filters,
    render_active_filter_bar,
    render_sidebar_filters,
)
from app.components.footer import render_footer
from app.components.metrics import render_kpi_card, render_severity_badge


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
    st.caption(
        "This page focuses on breach types and locations (for example, Network Server or Email). "
        "The KPIs and charts update as you change the date range and other filters."
    )

    df = get_data()

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

    analyzer = AttackVectorAnalyzer()
    vector_summary = analyzer.compute_vector_summary(filtered_df)
    evolution_df = analyzer.compute_vector_evolution(filtered_df)
    location_df = analyzer.compute_location_analysis(filtered_df)
    severity_matrix = analyzer.compute_vector_severity_matrix(filtered_df)

    # Per-page KPI row
    kpi_col1, kpi_col2, kpi_col3 = st.columns(3)

    if not vector_summary.empty:
        # Share of Hacking/IT Incident
        hacking_count = vector_summary.loc["Hacking/IT Incident", "count"] if "Hacking/IT Incident" in vector_summary.index else 0
        hacking_pct = (
            vector_summary.loc["Hacking/IT Incident", "percentage"]
            if "Hacking/IT Incident" in vector_summary.index
            else 0.0
        )
        with kpi_col1:
            render_kpi_card(
                title="Hacking/IT Incident Share",
                value=f"{hacking_pct:.1f}%",
                delta=f"{int(hacking_count):,} breaches",
            )

        # Most common breach type
        top_type = vector_summary.index[0]
        top_count = int(vector_summary.iloc[0]["count"])
        with kpi_col2:
            render_kpi_card(
                title="Most Common Breach Type",
                value=top_type,
                delta=f"{top_count:,} breaches",
            )

    if not location_df.empty:
        # Top breach location
        top_loc = location_df.index[0]
        top_loc_count = int(location_df.iloc[0]["count"])
        with kpi_col3:
            render_kpi_card(
                title="Top Breach Location",
                value=top_loc,
                delta=f"{top_loc_count:,} breaches",
            )

    render_active_filter_bar(time_meta, filter_state)

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

    # Narrative under breach-type evolution chart
    if not evolution_df.empty:
        years = sorted(evolution_df["year"].unique().tolist())
        if len(years) >= 2:
            start_year, end_year = years[0], years[-1]
            subset = evolution_df[evolution_df["year"].isin([start_year, end_year])]

            target_type = "Hacking/IT Incident"
            if target_type not in subset["breach_type"].unique():
                target_type = (
                    evolution_df["breach_type"].value_counts().idxmax()
                )

            start_row = subset[
                (subset["year"] == start_year)
                & (subset["breach_type"] == target_type)
            ]
            end_row = subset[
                (subset["year"] == end_year)
                & (subset["breach_type"] == target_type)
            ]

            if not start_row.empty and not end_row.empty:
                start_pct = float(start_row["percentage"].iloc[0])
                end_pct = float(end_row["percentage"].iloc[0])
                text = (
                    f"From **{start_year}** to **{end_year}**, **{target_type}** incidents changed "
                    f"from **{start_pct:.1f}%** to **{end_pct:.1f}%** of all breaches in this time window."
                )
                st.markdown(
                    f"<p style='color:#9CA3AF;font-size:0.85rem;'>{text}</p>",
                    unsafe_allow_html=True,
                )

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

