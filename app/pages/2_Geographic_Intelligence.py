from __future__ import annotations

import pandas as pd
import streamlit as st
from streamlit_folium import st_folium
import plotly.graph_objects as go
from pathlib import Path
import sys

from breachintel.utils.cache import load_data
from breachintel.utils.constants import STATE_ABBREVIATIONS
from breachintel.analysis.geographic import GeographicAnalyzer
from breachintel.visualization.maps import create_breach_heatmap, create_state_detail_map
from breachintel.visualization.charts import apply_theme, create_breach_type_area

# Ensure the project root (which contains the `app` package) is on sys.path so
# imports like `from app.components ...` work when the working directory is not
# the repository root (e.g., Streamlit Cloud).
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.components.filters import (
    configure_time_filters,
    render_active_filter_bar,
    render_sidebar_filters,
)
from app.components.metrics import render_kpi_card
from app.components.breach_detail import render_breach_detail_card
from app.components.footer import render_footer


st.set_page_config(
    page_title="BreachIntel — Geographic Intelligence",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data(ttl=3600)
def get_data() -> pd.DataFrame:
    return load_data()


def main() -> None:
    st.title("🗺️ Geographic Intelligence")

    # Clip folium map container to remove blank space below map (streamlit-folium #198, #213)
    st.markdown(
        "<style>[data-testid='stVerticalBlock'] > div:has(iframe) { max-height: 502px; overflow: hidden; }</style>",
        unsafe_allow_html=True,
    )

    df = get_data()

    # Sidebar: date range first, then breach/entity/geography (same as Home)
    st.sidebar.header("Filters")
    time_filtered_df, time_meta = configure_time_filters(df, in_sidebar=True)
    filtered_df, filter_state = render_sidebar_filters(time_filtered_df)

    # Contextual time summary + active filter chips
    analyzer = GeographicAnalyzer()
    state_summary = analyzer.compute_state_summary(filtered_df).reset_index()

    # Per-page KPI row
    kpi_col1, kpi_col2, kpi_col3 = st.columns(3)

    if not state_summary.empty:
        # Highest breach count state
        top_breach_idx = state_summary["breach_count"].idxmax()
        top_breach_row = state_summary.loc[top_breach_idx]
        with kpi_col1:
            render_kpi_card(
                title="Highest Breach Count State",
                value=f"{top_breach_row['state']} ({int(top_breach_row['breach_count']):,})",
            )

        # Highest per-capita state
        per_capita_df = state_summary.dropna(subset=["breaches_per_100k"])
        if not per_capita_df.empty:
            pc_idx = per_capita_df["breaches_per_100k"].idxmax()
            pc_row = per_capita_df.loc[pc_idx]
            with kpi_col2:
                render_kpi_card(
                    title="Highest Per-Capita State",
                    value=f"{pc_row['state']} ({pc_row['breaches_per_100k']:.2f} / 100k)",
                )

        # Number of states with breaches
        num_states = state_summary["state"].nunique()
        with kpi_col3:
            render_kpi_card(
                title="States With Breaches",
                value=f"{int(num_states):,}",
            )

    # Heatmap
    st.subheader("National Breach Heatmap")
    heatmap_map = create_breach_heatmap(state_summary)
    st_folium(heatmap_map, width=None, height=500)

    # Top states tables
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("#### Top 10 States by Total Breaches")
        st.markdown(
            "<p style='color:#9CA3AF;font-size:0.8rem;'>Breaches per 100,000 residents, "
            "based on census populations.</p>",
            unsafe_allow_html=True,
        )
        top_breaches = (
            state_summary.sort_values("breach_count", ascending=False)
            .head(10)
            .set_index("state")[["breach_count", "total_affected", "breaches_per_100k"]]
        )
        top_breaches_display = (
            top_breaches.rename(
                columns={
                    "breach_count": "Total Breaches",
                    "total_affected": "People Affected",
                    "breaches_per_100k": "Breaches per 100k",
                }
            ).rename_axis("State")
        )
        st.dataframe(
            top_breaches_display.style.format(
                {
                    "Total Breaches": "{:,.0f}",
                    "People Affected": "{:,.0f}",
                    "Breaches per 100k": "{:,.2f}",
                }
            ),
            use_container_width=True,
        )

    with col_right:
        st.markdown("#### Top 10 States by Per‑Capita Breach Rate")
        top_per_capita = (
            state_summary.sort_values("breaches_per_100k", ascending=False)
            .dropna(subset=["breaches_per_100k"])
            .head(10)
            .set_index("state")[["breach_count", "total_affected", "breaches_per_100k"]]
        )
        top_per_capita_display = (
            top_per_capita.rename(
                columns={
                    "breach_count": "Total Breaches",
                    "total_affected": "People Affected",
                    "breaches_per_100k": "Breaches per 100k",
                }
            ).rename_axis("State")
        )
        st.dataframe(
            top_per_capita_display.style.format(
                {
                    "Total Breaches": "{:,.0f}",
                    "People Affected": "{:,.0f}",
                    "Breaches per 100k": "{:,.2f}",
                }
            ),
            use_container_width=True,
        )

    st.markdown("---")

    # State detail section
    st.subheader("State Detail View")
    available_states = sorted(state_summary["state"].unique().tolist())
    selected_state = st.selectbox(
        "Select a state to inspect",
        options=available_states,
        index=0 if available_states else None,
    )

    if selected_state:
        state_df = filtered_df[filtered_df["state"] == selected_state]

        # Map state abbreviation to full name for human-readable headings.
        # STATE_ABBREVIATIONS maps many variants to USPS codes; build a reverse map.
        code_to_name = {}
        for full_or_code, code in STATE_ABBREVIATIONS.items():
            if len(full_or_code) > 2:
                code_to_name[code] = full_or_code.title()
        full_state_name = code_to_name.get(str(selected_state).upper(), str(selected_state).title())

        st.markdown(f"##### {full_state_name} Breach Map & Timeline")

        state_map = create_state_detail_map(state_df, selected_state)
        st_folium(state_map, width=None, height=400)

        # Timeline chart for selected state – aggregate breaches per year
        timeline_df = state_df.copy()
        timeline_df["breach_date"] = pd.to_datetime(
            timeline_df["breach_date"], errors="coerce"
        )
        timeline_df = timeline_df.dropna(subset=["breach_date"])
        timeline_df["year"] = timeline_df["breach_date"].dt.year

        yearly = (
            timeline_df.dropna(subset=["year"])
            .groupby("year")
            .size()
            .reset_index(name="count")
        )

        if yearly.empty:
            st.info(
                "No breaches with a valid year were found for this state and filter selection.",
                icon="ℹ️",
            )
        else:
            fig_timeline = go.Figure()
            fig_timeline.add_bar(
                x=yearly["year"].astype(str),
                y=yearly["count"],
                name="Breaches per Year",
                marker_color="#00E6B8",
                opacity=0.8,
            )
            fig_timeline.update_layout(
                title={
                    "text": f"Breaches Per Year in {full_state_name}",
                    "x": 0.5,
                    "xanchor": "center",
                },
                xaxis_title="Year",
                yaxis_title="Number of Breaches",
                hovermode="x unified",
            )
            fig_timeline.update_yaxes(rangemode="tozero")
            fig_timeline.update_xaxes(type="category")
            fig_timeline = apply_theme(fig_timeline)
            st.plotly_chart(fig_timeline, use_container_width=True)

        # Breach type composition over time for the state
        st.markdown(f"##### Breach Type Evolution in {full_state_name}")
        bt_df = state_df.copy()
        bt_df["date"] = bt_df["breach_date"]
        fig_bt = create_breach_type_area(bt_df)
        st.plotly_chart(fig_bt, use_container_width=True)

        # Drill-down: top breaches in this state
        st.markdown(f"##### Top Breaches in {full_state_name}")
        if state_df.empty:
            st.info(
                "No breaches match this state with the current filters. "
                "Try broadening the filters or selecting a different state.",
                icon="ℹ️",
            )
        else:
            ranked = state_df.sort_values(
                "individuals_affected", ascending=False
            )
            top_state = ranked[
                ["entity_name", "individuals_affected", "breach_type", "breach_location"]
            ].head(10)
            top_state_display = top_state.rename(
                columns={
                    "entity_name": "Healthcare Entity",
                    "individuals_affected": "People Affected",
                    "breach_type": "Breach Cause",
                    "breach_location": "Compromised System",
                }
            )

            st.dataframe(
                top_state_display.style.format({"People Affected": "{:,.0f}"}),
                use_container_width=True,
                hide_index=True,
            )

            st.markdown("**Inspect entity in this state**")
            selected_entity_state = st.selectbox(
                "Choose an entity",
                options=top_state["entity_name"].tolist(),
                key="geo_entity_inspect",
                label_visibility="collapsed",
            )
            if selected_entity_state:
                entity_rows_state = ranked[
                    ranked["entity_name"] == selected_entity_state
                ]
                if not entity_rows_state.empty:
                    idx_state = entity_rows_state["individuals_affected"].idxmax()
                    record_state = entity_rows_state.loc[idx_state]
                    # Render the detail content without the outer framed box.
                    render_breach_detail_card(
                        record_state,
                        title="State Breach Detail",
                        show_frame=False,
                    )

    render_footer()


if __name__ == "__main__":
    main()

