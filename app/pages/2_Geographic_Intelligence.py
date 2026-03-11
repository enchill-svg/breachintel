from __future__ import annotations

import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

from breachintel.utils.cache import load_data
from breachintel.analysis.geographic import GeographicAnalyzer
from breachintel.visualization.maps import create_breach_heatmap, create_state_detail_map
from breachintel.visualization.charts import create_overview_chart, create_breach_type_area

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
    st.caption(
        "This page compares states by total breach counts and per‑capita rates. Use the state "
        "selector and filters to drill into a single state’s map and timelines."
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

    render_active_filter_bar(time_meta, filter_state)

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
        st.dataframe(
            top_breaches.style.format(
                {
                    "breach_count": "{:,.0f}",
                    "total_affected": "{:,.0f}",
                    "breaches_per_100k": "{:,.2f}",
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
        st.dataframe(
            top_per_capita.style.format(
                {
                    "breach_count": "{:,.0f}",
                    "total_affected": "{:,.0f}",
                    "breaches_per_100k": "{:,.2f}",
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

        st.markdown(f"##### {selected_state} Breach Map & Timeline")

        state_map = create_state_detail_map(state_df, selected_state)
        st_folium(state_map, width=None, height=400)

        # Timeline chart for selected state
        timeline_df = state_df.copy()
        timeline_df = timeline_df.rename(columns={"breach_date": "date"})
        fig_timeline = create_overview_chart(timeline_df)
        st.plotly_chart(fig_timeline, use_container_width=True)

        # Breach type composition over time for the state
        st.markdown("##### Breach Type Evolution in State")
        bt_df = state_df.copy()
        bt_df["date"] = bt_df["breach_date"]
        fig_bt = create_breach_type_area(bt_df)
        st.plotly_chart(fig_bt, use_container_width=True)

        # Drill-down: top breaches in this state
        st.markdown("##### Top Breaches in State")
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

            left_col, right_col = st.columns([2, 1])

            with left_col:
                st.dataframe(
                    top_state.style.format(
                        {"individuals_affected": "{:,.0f}"}
                    ),
                    use_container_width=True,
                )

            with right_col:
                selected_entity_state = st.selectbox(
                    "Inspect entity in this state",
                    options=top_state["entity_name"].tolist(),
                    key="geo_entity_inspect",
                )
                if selected_entity_state:
                    entity_rows_state = ranked[
                        ranked["entity_name"] == selected_entity_state
                    ]
                    if not entity_rows_state.empty:
                        idx_state = entity_rows_state["individuals_affected"].idxmax()
                        record_state = entity_rows_state.loc[idx_state]
                        render_breach_detail_card(
                            record_state, title="State Breach Detail"
                        )

    render_footer()


if __name__ == "__main__":
    main()

