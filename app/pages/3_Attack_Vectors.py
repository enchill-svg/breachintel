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

    df = get_data()

    # Sidebar: date range first, then breach/entity/geography (same as Home)
    st.sidebar.header("Filters")
    time_filtered_df, time_meta = configure_time_filters(df, in_sidebar=True)
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
    # Round key metrics for readability
    summary_display["avg_affected"] = summary_display["avg_affected"].round(0)
    summary_display["median_affected"] = summary_display["median_affected"].round(0)
    summary_display["percentage"] = summary_display["percentage"].round(1)
    # Move breach_type out of the index and apply human-readable column names
    summary_display = (
        summary_display.reset_index()
        .rename(
            columns={
                "breach_type": "Breach Type",
                "count": "Count",
                "total_affected": "Total Affected",
                "avg_affected": "Avg Affected",
                "median_affected": "Median Affected",
                "max_affected": "Max Affected",
                "percentage": "Percentage",
            }
        )
    )
    # Ensure Percentage column stays numeric so Styler format strings work reliably
    summary_display["Percentage"] = pd.to_numeric(
        summary_display["Percentage"], errors="coerce"
    )
    st.dataframe(
        summary_display.style.format(
            {
                "Count": "{:,.0f}",
                "Total Affected": "{:,.0f}",
                "Avg Affected": "{:,.0f}",
                "Median Affected": "{:,.0f}",
                "Max Affected": "{:,.0f}",
                "Percentage": "{:.1f}%",
            }
        ),
        use_container_width=True,
        hide_index=True,
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
                # Render as plain markdown so the bold formatting displays correctly.
                st.markdown(
                    f"From **{start_year}** to **{end_year}**, **{target_type}** incidents changed "
                    f"from **{start_pct:.1f}%** to **{end_pct:.1f}%** of all breaches in this time window.",
                )

    st.markdown("---")

    # Breach location analysis - horizontal bar chart
    st.subheader("Where Are Breaches Occurring?")
    location_summary = (
        location_df.reset_index()
        .rename(columns={"breach_location": "entity_type", "count": "breach_count"})
    )
    fig_location = create_entity_comparison(location_summary)
    # Replace database-style labels with plain language for this chart.
    # Clear the Plotly title to avoid duplicating the Streamlit subheader.
    fig_location.update_layout(
        title_text="",
        xaxis_title="Number of Breaches",
        yaxis_title="Breach Location",
        legend_title_text="Location",
    )
    st.plotly_chart(fig_location, use_container_width=True)

    # Severity matrix as styled table
    st.subheader("Severity Matrix (Breach Type × Severity)")
    severity_display = severity_matrix.copy().reset_index()
    severity_display = severity_display.rename(columns={"breach_type": "Breach Type"})

    # Reorder severity columns to logical order: Low, Medium, High, Critical, Total
    desired_cols = ["Low", "Medium", "High", "Critical", "Total"]
    ordered_severity_cols = [c for c in desired_cols if c in severity_display.columns]
    # Preserve "Breach Type" as the first column, followed by ordered severity columns
    other_cols = [
        c
        for c in severity_display.columns
        if c not in ordered_severity_cols and c != "Breach Type"
    ]
    new_column_order = ["Breach Type"] + ordered_severity_cols + other_cols
    severity_display = severity_display[new_column_order]

    # Only apply numeric formatting to numeric columns to avoid ValueError on string columns
    numeric_cols = [
        c for c in severity_display.columns if c != "Breach Type"
    ]
    format_map = {c: "{:,.0f}" for c in numeric_cols}
    st.dataframe(
        severity_display.style.format(format_map),
        use_container_width=True,
        hide_index=True,
    )

    render_footer()


if __name__ == "__main__":
    main()

