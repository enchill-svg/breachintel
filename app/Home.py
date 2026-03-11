from __future__ import annotations

from typing import Dict

import pandas as pd
import streamlit as st

from breachintel.utils.cache import load_data
from breachintel.analysis.trends import TrendAnalyzer
from breachintel.visualization.charts import create_overview_chart
from breachintel.utils.constants import COLORS

from app.components.metrics import render_kpi_card, render_severity_badge  # noqa: F401
from app.components.filters import (
    configure_time_filters,
    render_active_filter_bar,
    render_sidebar_filters,
)
from app.components.breach_detail import render_breach_detail_card
from app.components.footer import render_footer


st.set_page_config(
    page_title="BreachIntel — Healthcare Breach Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _load_custom_css() -> None:
    css_path = "app/assets/custom.css"
    try:
        with open(css_path, "r", encoding="utf-8") as f:
            css = f.read()
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        # Fail silently if custom CSS is missing
        pass


@st.cache_data(ttl=3600)
def get_data():
    return load_data()


@st.cache_data
def compute_headline_metrics(df) -> Dict[str, int]:
    analyzer = TrendAnalyzer()
    return analyzer.compute_headline_metrics(df)


def format_int(value) -> str:
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return "N/A"


def format_big_number(value) -> str:
    """
    Format large numbers using K/M abbreviations with one decimal place.
    Examples:
    - 943700000 -> '943.7M'
    - 514100    -> '514.1K'
    """
    try:
        n = float(value)
    except (TypeError, ValueError):
        return "N/A"

    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return format_int(n)


def main() -> None:
    _load_custom_css()

    df = get_data()
    metrics = compute_headline_metrics(df)

    total_breaches = metrics["total_breaches"]
    total_individuals = metrics["total_individuals_affected"]
    states_affected = metrics["states_affected"]
    current_year_breaches = metrics["current_year_breaches"]
    yoy_change_pct = metrics["yoy_change_pct"]
    avg_breaches_per_month = metrics["avg_breaches_per_month"]

    # Business associate: sum individuals_affected where business_associate contains 'Yes'
    if "business_associate" in df.columns and "individuals_affected" in df.columns:
        ba_mask = (
            df["business_associate"]
            .astype(str)
            .str.upper()
            .str.contains("YES", na=False)
        )
        individuals_ba = int(df.loc[ba_mask, "individuals_affected"].sum())
    else:
        individuals_ba = 0

    # Top-of-page time controls
    time_filtered_df, time_meta = configure_time_filters(df)

    # Sidebar filters (breach type, entity type, geography)
    filtered_df, filter_state = render_sidebar_filters(time_filtered_df)

    # Header
    header_html = f"""
    <div style="text-align:center; padding: 1rem 0 0.5rem 0;">
        <h1 style="margin-bottom:0.25rem; color:{COLORS['primary']};">
            🛡️ BreachIntel
        </h1>
        <p style="margin:0; font-size:1.1rem; color:{COLORS['text_secondary']};">
            Healthcare Cybersecurity Breach Intelligence Platform
        </p>
        <p style="margin-top:0.25rem; font-size:0.95rem; color:{COLORS['text_secondary']};">
            Tracking <strong>{format_int(total_breaches)}</strong> breaches impacting
            <strong>{format_int(total_individuals)}</strong> individuals across
            <strong>{format_int(states_affected)}</strong> states.
        </p>
    </div>
    """
    st.markdown(header_html, unsafe_allow_html=True)
    st.caption(
        "Use the date range and filters to see how overall breach volume and affected "
        "individuals change over time. The largest single breach and headline KPIs always "
        "respect your current time window."
    )

    # KPI row
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric(
            label="Total Breaches",
            value=format_int(total_breaches),
        )
    with col2:
        if yoy_change_pct is not None:
            delta_value = f"{yoy_change_pct:+.1f}% vs same period prior year"
        else:
            delta_value = "N/A"
        st.metric(
            label="Current Year Breaches",
            value=format_int(current_year_breaches),
            delta=delta_value,
            delta_color="inverse",
        )
    with col3:
        st.metric(
            label="Total Individuals Affected",
            value=format_big_number(total_individuals),
        )
    with col4:
        st.metric(
            label="Avg Monthly Breaches",
            value=f"{avg_breaches_per_month:.1f}",
        )
    with col5:
        st.metric(
            label="Individuals Affected (BA Present)",
            value=format_big_number(individuals_ba),
        )

    # Contextual time summary for the current view (based on breach submission dates)
    if time_meta:
        start = time_meta.get("start_date")
        end = time_meta.get("end_date")
        total_incidents = time_meta.get("total_incidents", len(filtered_df))
        st.caption(
            f"Showing breaches from **{start}** to **{end}** "
            f"({format_int(total_incidents)} incidents before additional filters)."
        )

    # Active filter chips
    render_active_filter_bar(time_meta, filter_state)

    st.divider()

    # Overview chart
    # For charts we expect a 'date' column; TrendAnalyzer uses 'breach_date'
    chart_df = filtered_df.rename(columns={"breach_date": "date"})
    fig_overview = create_overview_chart(chart_df)
    st.plotly_chart(fig_overview, use_container_width=True)

    # Largest breach detail card
    largest_entity = metrics["largest_breach_entity"]
    largest_count = metrics["largest_breach_count"]
    if largest_entity and largest_count and "entity_name" in df.columns:
        largest_rows = df[df["entity_name"] == largest_entity]
        if "individuals_affected" in largest_rows.columns:
            largest_rows = largest_rows.sort_values(
                "individuals_affected", ascending=False
            )
        record = largest_rows.iloc[0]
        render_breach_detail_card(record, title="Largest Single Breach")

    # Navigation hint
    st.markdown(
        """
        <div style="margin-top:1.5rem; text-align:center; font-size:0.9rem; color:#9CA3AF;">
            Explore detailed analytics via the sidebar: breach types, entities, geography, and forecasts.
        </div>
        """,
        unsafe_allow_html=True,
    )

    render_footer()


if __name__ == "__main__":
    main()

