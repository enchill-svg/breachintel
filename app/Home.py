from __future__ import annotations

from typing import Dict
from pathlib import Path
import sys

import pandas as pd
import streamlit as st

from breachintel.utils.cache import load_data
from breachintel.analysis.trends import TrendAnalyzer
from breachintel.visualization.charts import create_overview_chart
from breachintel.utils.constants import COLORS

# Ensure the project root (which contains the `app` package) is on sys.path so
# imports like `from app.components ...` work in environments where the working
# directory is not the repository root (e.g., Streamlit Cloud).
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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

    # Sidebar filters including time controls
    st.sidebar.header("Filters")
    time_filtered_df, time_meta = configure_time_filters(df, in_sidebar=True)
    filtered_df, filter_state = render_sidebar_filters(time_filtered_df)

    # Header
    header_html = f"""
    <div style="
        padding: 0.6rem 1.0rem 0.75rem 1.0rem;
        margin-top: 0.2rem;
        margin-bottom: 0.75rem;
        background-color: {COLORS['bg_card']};
        border-radius: 0.75rem;
        border: 1px solid rgba(148,163,184,0.4);
    ">
        <h1 style="margin:0; color:{COLORS['primary']}; font-size:1.6rem;">
            🛡️ BreachIntel
        </h1>
        <p style="margin:0.15rem 0 0 0; font-size:1.0rem; color:{COLORS['text_secondary']};">
            Healthcare Cybersecurity Breach Intelligence Platform
        </p>
        <p style="margin:0.25rem 0 0 0; font-size:0.95rem; color:{COLORS['text_secondary']};">
            Tracking <strong>{format_int(total_breaches)}</strong> reported healthcare data breaches
            affecting over <strong>{format_big_number(total_individuals)}</strong> patient records across
            all 50 U.S. states and territories.
        </p>
    </div>
    """
    st.markdown(header_html, unsafe_allow_html=True)
    # KPI row (4 equally spaced columns)
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="Total Breaches",
            value=format_int(total_breaches),
        )
    with col2:
        if yoy_change_pct is not None:
            delta_value = f"{yoy_change_pct:+.1f}% vs prior year"
        else:
            delta_value = "N/A"
        # Be explicit about the current calendar year in the label
        current_year = pd.to_datetime("today").year
        st.metric(
            label=f"Breaches in {current_year}",
            value=format_int(current_year_breaches),
            delta=delta_value,
            delta_color="inverse",
        )
    with col3:
        st.metric(
            label="Patient Records Exposed",
            value=format_big_number(total_individuals),
        )
    with col4:
        st.metric(
            label="Exposed via Third Parties",
            value=format_big_number(individuals_ba),
        )

    # Impact story for non-technical readers
    try:
        org_count = int(df["entity_name"].nunique()) if "entity_name" in df.columns else 0
    except Exception:
        org_count = 0
    story_text = (
        f"Since 2009, over {format_int(org_count)} healthcare organizations have reported data breaches "
        f"to the U.S. Department of Health and Human Services, exposing approximately "
        f"{format_big_number(total_individuals)} patient records. With a U.S. population of roughly "
        "330 million, this means the average American's health data has been compromised nearly 3 times."
    )
    # Constrain story width for readability using a max-width container, no italics
    st.markdown(
        f"""
        <div style="max-width: 52rem; font-size:0.9rem; color:{COLORS['text_secondary']}; margin-top:0.35rem;">
            {story_text}
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Data status line and largest-breach callout
    max_date = None
    if "breach_date" in df.columns:
        max_ts = pd.to_datetime(df["breach_date"], errors="coerce").max()
        if pd.notna(max_ts):
            max_date = max_ts.date()

    if time_meta:
        start = time_meta.get("start_date")
        end = time_meta.get("end_date")
        total_incidents = time_meta.get("total_incidents", len(filtered_df))
        status_parts = []
        if max_date:
            status_parts.append(f"Data through: {max_date}")
        if start and end:
            status_parts.append(
                f"Showing: {start} – {end} ({format_int(total_incidents)} incidents)"
            )
        if status_parts:
            st.caption(" | ".join(status_parts))

    # Largest breach summary callout
    largest_entity = metrics["largest_breach_entity"]
    largest_count = metrics["largest_breach_count"]
    largest_record = None
    if largest_entity and largest_count and "entity_name" in df.columns:
        largest_rows = df[df["entity_name"] == largest_entity]
        if "individuals_affected" in largest_rows.columns:
            largest_rows = largest_rows.sort_values(
                "individuals_affected", ascending=False
            )
        if not largest_rows.empty:
            largest_record = largest_rows.iloc[0]
        else:
            st.info("No breach details available for the current filter selection.")

    # Spacer before main chart
    st.divider()
    chart_df = filtered_df.rename(columns={"breach_date": "date"})
    fig_overview = create_overview_chart(chart_df)

    # Descriptive sentence should appear between the title and the rendered chart
    st.caption(
        "Each bar shows the number of reported healthcare data breaches per month. "
        "The purple line tracks the 12-month rolling average."
    )

    st.plotly_chart(fig_overview, use_container_width=True)

    # Detailed breach card beneath chart
    if largest_record is not None:
        render_breach_detail_card(largest_record, title="Largest Single Breach")

    render_footer()


if __name__ == "__main__":
    main()

