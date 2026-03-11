from __future__ import annotations

import pandas as pd
import streamlit as st

from breachintel.utils.cache import load_data
from breachintel.analysis.entity_profiling import EntityProfiler
from breachintel.visualization.charts import (
    create_entity_comparison,
    create_breach_type_area,
)

from app.components.filters import (
    configure_time_filters,
    render_active_filter_bar,
    render_sidebar_filters,
)
from app.components.footer import render_footer
from app.components.metrics import render_kpi_card
from app.components.breach_detail import render_breach_detail_card


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
        "Compare breach exposure across healthcare entity types and identify repeat offenders "
        "driving systemic risk. Use the filters and time controls to focus on particular entity "
        "types or periods."
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

    profiler = EntityProfiler()
    entity_summary = profiler.compute_entity_summary(filtered_df)
    entity_trend = profiler.compute_entity_trend(filtered_df)
    repeat_offenders = profiler.find_most_breached_entities(filtered_df, top_n=20)

    # Per-page KPI row
    kpi_col1, kpi_col2, kpi_col3 = st.columns(3)

    # Number of entities with 3+ breaches in current window
    if not repeat_offenders.empty:
        high_repeat = repeat_offenders[repeat_offenders["breach_count"] >= 3]
        num_high_repeat = int(high_repeat.shape[0])
    else:
        num_high_repeat = 0
    with kpi_col1:
        render_kpi_card(
            title="Entities With 3+ Breaches",
            value=f"{num_high_repeat:,}",
        )

    # Top repeat-offender entity
    if not repeat_offenders.empty:
        top_entity_row = repeat_offenders.sort_values(
            ["breach_count", "total_affected"], ascending=[False, False]
        ).iloc[0]
        with kpi_col2:
            render_kpi_card(
                title="Top Repeat-Offender Entity",
                value=str(top_entity_row.name),
                delta=f"{int(top_entity_row['breach_count']):,} breaches",
            )

    # Share of breaches involving business associates
    if "business_associate" in filtered_df.columns:
        ba_mask = (
            filtered_df["business_associate"]
            .astype(str)
            .str.upper()
            .str.contains("YES", na=False)
        )
        total_breaches = int(len(filtered_df))
        ba_count = int(ba_mask.sum())
        ba_pct = (ba_count / total_breaches * 100.0) if total_breaches > 0 else 0.0
        with kpi_col3:
            render_kpi_card(
                title="Breaches With Business Associate",
                value=f"{ba_pct:.1f}%",
                delta=f"{ba_count:,} of {total_breaches:,}",
            )

    render_active_filter_bar(time_meta, filter_state)

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

    # Repeat offenders section with drill-down
    st.subheader("Repeat Offenders")
    repeat_display = repeat_offenders.reset_index()
    repeat_display = repeat_display[
        ["entity_name", "breach_count", "total_affected", "entity_type", "state"]
    ]

    left_col, right_col = st.columns([2, 1])

    with left_col:
        st.dataframe(
            repeat_display.style.format(
                {
                    "breach_count": "{:,.0f}",
                    "total_affected": "{:,.0f}",
                }
            ),
            use_container_width=True,
        )

    with right_col:
        if not repeat_display.empty:
            selected_entity = st.selectbox(
                "Inspect entity",
                options=repeat_display["entity_name"].tolist(),
            )
            if selected_entity:
                # Use the largest breach for this entity in the current filtered window
                entity_rows = filtered_df[filtered_df["entity_name"] == selected_entity]
                if not entity_rows.empty:
                    idx = entity_rows["individuals_affected"].idxmax()
                    record = entity_rows.loc[idx]
                    render_breach_detail_card(
                        record,
                        title="Entity Breach Detail",
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

