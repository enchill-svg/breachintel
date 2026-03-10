from __future__ import annotations

from typing import Dict

import streamlit as st

from breachintel.utils.cache import load_data
from breachintel.analysis.trends import TrendAnalyzer
from breachintel.visualization.charts import create_overview_chart
from breachintel.utils.constants import COLORS

from .components.metrics import render_kpi_card, render_severity_badge  # noqa: F401
from .components.filters import render_sidebar_filters
from .components.footer import render_footer


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

    # Sidebar filters
    filtered_df = render_sidebar_filters(df)

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

    # KPI row
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="Total Breaches",
            value=format_int(total_breaches),
        )
    with col2:
        delta_value = f"{yoy_change_pct}%" if yoy_change_pct is not None else "N/A"
        st.metric(
            label="Current Year Breaches",
            value=format_int(current_year_breaches),
            delta=delta_value,
            delta_color="inverse",
        )
    with col3:
        st.metric(
            label="Total Individuals Affected",
            value=format_int(total_individuals),
        )
    with col4:
        st.metric(
            label="Avg Monthly Breaches",
            value=f"{avg_breaches_per_month:.1f}",
        )

    st.divider()

    # Overview chart
    st.subheader("Breach Activity Over Time")

    # For charts we expect a 'date' column; TrendAnalyzer uses 'breach_date'
    chart_df = filtered_df.rename(columns={"breach_date": "date"})
    fig_overview = create_overview_chart(chart_df)
    st.plotly_chart(fig_overview, use_container_width=True)

    # Info callout for largest breach
    largest_entity = metrics["largest_breach_entity"]
    largest_count = metrics["largest_breach_count"]
    if largest_entity and largest_count:
        st.info(
            f"**Largest single breach**: {largest_entity} with "
            f"{format_int(largest_count)} individuals affected.",
            icon="🔥",
        )

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

