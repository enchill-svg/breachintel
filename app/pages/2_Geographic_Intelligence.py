from __future__ import annotations

import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

from breachintel.utils.cache import load_data
from breachintel.analysis.geographic import GeographicAnalyzer
from breachintel.visualization.maps import create_breach_heatmap, create_state_detail_map
from breachintel.visualization.charts import create_overview_chart, create_breach_type_area

from app.components.filters import render_sidebar_filters
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
        "Understand how healthcare breaches are distributed across states and how "
        "per‑capita risk differs from raw incident counts."
    )

    df = get_data()
    filtered_df = render_sidebar_filters(df)

    analyzer = GeographicAnalyzer()
    state_summary = analyzer.compute_state_summary(filtered_df).reset_index()

    # Heatmap
    st.subheader("National Breach Heatmap")
    heatmap_map = create_breach_heatmap(state_summary)
    st_folium(heatmap_map, width=None, height=500)

    # Top states tables
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("#### Top 10 States by Total Breaches")
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

    render_footer()


if __name__ == "__main__":
    main()

