from __future__ import annotations

from typing import Dict
from pathlib import Path
import sys

import pandas as pd
import streamlit as st

from breachintel.utils.cache import load_data
from breachintel.analysis.trends import TrendAnalyzer
from breachintel.visualization.charts import (
    create_overview_chart,
    create_yoy_growth,
    create_breach_type_area,
)

# Ensure the project root (which contains the `app` package) is on sys.path so
# imports like `from app.components ...` work in environments where the working
# directory is not the repository root (e.g., Streamlit Cloud).
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
    page_title="BreachIntel — Trend Analysis",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data(ttl=3600)
def get_data() -> pd.DataFrame:
    return load_data()


def compute_trend_data(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    analyzer = TrendAnalyzer()
    monthly = analyzer.compute_monthly_trends(df)
    yearly = analyzer.compute_yearly_trends(df)
    return {"monthly": monthly, "yearly": yearly}


def _compute_yearly_with_year_mode(
    df: pd.DataFrame,
    year_mode: str,
) -> pd.DataFrame:
    """
    Compute yearly trends using either calendar year or fiscal year (Oct–Sep).

    Calendar: identical to TrendAnalyzer.compute_yearly_trends.
    Fiscal: year is based on breach_date shifted +3 months, so Oct–Dec roll into next year.
    """
    df_dt = df.copy()
    df_dt["breach_date"] = pd.to_datetime(df_dt["breach_date"], errors="coerce")
    if df_dt["breach_date"].isna().all():
        return pd.DataFrame()

    if year_mode == "Fiscal (Oct–Sep)":
        shifted = df_dt["breach_date"] + pd.DateOffset(months=3)
        df_dt["year"] = shifted.dt.year
    else:
        df_dt["year"] = df_dt["breach_date"].dt.year

    grouped = df_dt.groupby("year")

    yearly = grouped.agg(
        breach_count=("breach_date", "size"),
        total_affected=("individuals_affected", "sum"),
        avg_affected=("individuals_affected", "mean"),
        median_affected=("individuals_affected", "median"),
        unique_entities=("entity_name", "nunique"),
        unique_states=("state", "nunique"),
    ).sort_index()

    yearly["yoy_count_change"] = yearly["breach_count"].pct_change() * 100.0
    yearly["yoy_affected_change"] = yearly["total_affected"].pct_change() * 100.0

    return yearly


def main() -> None:
    st.title("📈 Trend Analysis")

    df = get_data()

    # Sidebar: date range and filters (time first, then breach/entity/geography)
    st.sidebar.header("Filters")
    time_filtered_df, time_meta = configure_time_filters(df, in_sidebar=True)
    filtered_df, filter_state = render_sidebar_filters(time_filtered_df)

    # Contextual time summary
    if time_meta:
        start = time_meta.get("start_date")
        end = time_meta.get("end_date")
        total_incidents = time_meta.get("total_incidents", len(time_filtered_df))
        st.caption(
            f"Showing breaches from {start} to {end} "
            f"({int(total_incidents):,} incidents before additional filters)."
        )

    trend_data = compute_trend_data(filtered_df)
    monthly = trend_data["monthly"]

    # Yearly trends (calendar year by default)
    yearly = _compute_yearly_with_year_mode(filtered_df, year_mode="Calendar year")

    # Per-page KPI row
    kpi_col1, kpi_col2, kpi_col3 = st.columns(3)

    # Max monthly breaches in active window
    if not monthly.empty:
        max_row = monthly["breach_count"].idxmax()
        max_count = int(monthly.loc[max_row, "breach_count"])
        max_date = max_row.strftime("%Y-%m")
        with kpi_col1:
            render_kpi_card(
                title="Max Monthly Breaches",
                value=f"{max_count:,}",
                delta=f"Peak month: {max_date}",
            )

    # Most incident-heavy year
    if not yearly.empty:
        peak_year = int(yearly["breach_count"].idxmax())
        peak_count = int(yearly.loc[peak_year, "breach_count"])
        yoy = yearly.loc[peak_year, "yoy_count_change"]
        yoy_label = f"{yoy:+.1f}% YoY" if pd.notna(yoy) else "YoY change: N/A"
        with kpi_col2:
            render_kpi_card(
                title="Most Incident-Heavy Year",
                value=f"{peak_year} ({peak_count:,})",
                delta=yoy_label,
            )

    # Largest single YoY spike
    if "yoy_count_change" in yearly.columns and yearly["yoy_count_change"].notna().any():
        spikes = yearly["yoy_count_change"].dropna()
        spike_year = int(spikes.idxmax())
        spike_val = float(spikes.loc[spike_year])
        with kpi_col3:
            render_kpi_card(
                title="Largest YoY Spike",
                value=f"{spike_val:+.1f}%",
                delta=f"Year: {spike_year}",
                color="danger" if spike_val >= 0 else "success",
            )

    # Global group-by selector for trend views
    group_by = st.radio(
        "Group by",
        options=["Month", "Year"],
        index=0,
        horizontal=True,
        key="trend_group_by",
    )

    if group_by == "Month":
        fig_overview = create_overview_chart(filtered_df)
        st.plotly_chart(fig_overview, use_container_width=True)

        # Narrative summary under monthly chart
        if not monthly.empty and len(monthly) >= 12:
            # Compute average monthly breaches for the first and last *complete* years
            monthly_index = monthly.index.to_period("M")
            years = sorted(set(monthly_index.year))
            complete_years = []
            for y in years:
                mask = monthly_index.year == y
                if mask.sum() == 12:
                    complete_years.append(y)

            if len(complete_years) >= 2:
                first_year = complete_years[0]
                last_year = complete_years[-1]
                first_avg = float(
                    monthly[monthly_index.year == first_year]["breach_count"].mean()
                )
                last_avg = float(
                    monthly[monthly_index.year == last_year]["breach_count"].mean()
                )

                text: str
                if (
                    "yoy_count_change" in yearly.columns
                    and yearly["yoy_count_change"].notna().any()
                ):
                    spikes = yearly["yoy_count_change"].dropna()
                    spike_year = int(spikes.idxmax())
                    spike_val = float(spikes.loc[spike_year])
                    text = (
                        f"Between <strong>{first_year}</strong> and <strong>{last_year}</strong>, the "
                        f"average number of monthly breaches shifted from "
                        f"<strong>{first_avg:.1f}</strong> to <strong>{last_avg:.1f}</strong> incidents, "
                        f"with the largest single-year spike in <strong>{spike_year}</strong> "
                        f"({spike_val:+.1f}% YoY)."
                    )
                else:
                    text = (
                        f"Between <strong>{first_year}</strong> and <strong>{last_year}</strong>, the "
                        f"average number of monthly breaches shifted from "
                        f"<strong>{first_avg:.1f}</strong> to <strong>{last_avg:.1f}</strong> incidents."
                    )
            else:
                # Fallback to original point-to-point logic if we cannot find two full years
                first_idx = monthly.index.min()
                last_idx = monthly.index.max()
                first_cnt = int(monthly.loc[first_idx, "breach_count"])
                last_cnt = int(monthly.loc[last_idx, "breach_count"])
                first_year = first_idx.year
                last_year = last_idx.year
                text = (
                    f"From <strong>{first_year}</strong> to <strong>{last_year}</strong>, monthly breaches changed from "
                    f"<strong>{first_cnt}</strong> to <strong>{last_cnt}</strong> incidents."
                )

            st.markdown(
                f"<p style='color:#9CA3AF;font-size:0.85rem;'>{text}</p>",
                unsafe_allow_html=True,
            )

        # Inflection points: 3 most significant by absolute change, with at least one from 2017+
        analyzer = TrendAnalyzer()
        inflections = analyzer.detect_inflection_points(monthly, max_points=15)
        cutoff_2017 = pd.Timestamp("2017-01-01")
        recent = [p for p in inflections if p["date"] >= cutoff_2017]
        if not recent:
            display_inflections = inflections[:3]
        elif any(p["date"] >= cutoff_2017 for p in inflections[:3]):
            display_inflections = inflections[:3]
        else:
            best_recent = recent[0]
            display_inflections = inflections[:2] + [best_recent]
        if display_inflections:
            for point in display_inflections:
                dt = point["date"]
                ma_val = point["breach_count_ma"]
                direction = point["direction"]
                direction_label = "upward" if direction == "up" else "downward"
                st.info(
                    f"Trend inflection on **{dt.date()}**: 12-month average "
                    f"at **{ma_val:.1f}** breaches, turning **{direction_label}**.",
                    icon="🔍",
                )
        else:
            st.info(
                "The frequency of healthcare breaches remains consistently high, with no significant "
                "downward trend observed over the past year.",
                icon="ℹ️",
            )

        # Drill-down: inspect a specific month
        if not monthly.empty:
            st.subheader("Drill into a Month")

            # Build list of available periods as YYYY-MM strings
            periods = [idx.strftime("%Y-%m") for idx in monthly.index]
            default_period = periods[-1] if periods else None

            selected_period = st.selectbox(
                "Select a month to inspect",
                options=periods,
                index=len(periods) - 1 if periods else 0,
                key="trend_inspect_period",
            ) if periods else None

            if selected_period:
                year, month = map(int, selected_period.split("-"))
                month_mask = (
                    pd.to_datetime(filtered_df["breach_date"], errors="coerce").dt.year.eq(
                        year
                    )
                    & pd.to_datetime(
                        filtered_df["breach_date"], errors="coerce"
                    ).dt.month.eq(month)
                )
                month_df = filtered_df.loc[month_mask].copy()

                if month_df.empty:
                    st.info(
                        "No breaches match this month with the current filters. "
                        "Try selecting a different month or broadening the filters.",
                        icon="ℹ️",
                    )
                else:
                    month_df = month_df.sort_values(
                        "individuals_affected", ascending=False
                    )
                    top_display = month_df[
                        [
                            "entity_name",
                            "individuals_affected",
                            "breach_type",
                            "state",
                        ]
                    ].head(10)
                    # Present user-friendly column headers for non-technical readers
                    top_display = top_display.rename(
                        columns={
                            "entity_name": "Healthcare Entity",
                            "individuals_affected": "People Affected",
                            "breach_type": "Breach Cause",
                            "state": "State",
                        }
                    )

                    # Hide the DataFrame index; users don't need raw row numbers
                    st.dataframe(
                        top_display.style.format(
                            {"People Affected": "{:,.0f}"}
                        ),
                        use_container_width=True,
                        hide_index=True,
                    )

                    # Breach detail card below the table at full width for better readability
                    selected_entity = st.selectbox(
                        "Inspect entity in this month",
                        options=top_display["Healthcare Entity"].tolist(),
                        key="trend_entity_inspect",
                    )
                    if selected_entity:
                        entity_rows = month_df[
                            month_df["entity_name"] == selected_entity
                        ]
                        if not entity_rows.empty:
                            idx = entity_rows["individuals_affected"].idxmax()
                            record = entity_rows.loc[idx]
                            render_breach_detail_card(
                                record,
                                title="Monthly Breach Detail",
                                show_frame=False,
                            )
    else:
        st.subheader("Yearly Breach Metrics")

        # Restrict YoY analysis to complete historical years (exclude the latest incomplete year)
        # by aligning with the monthly index used elsewhere.
        yearly_complete = yearly.copy()
        if not monthly.empty:
            monthly_index = monthly.index.to_period("M")
            years = sorted(set(monthly_index.year))
            complete_years = []
            for y in years:
                mask = monthly_index.year == y
                if mask.sum() == 12:
                    complete_years.append(y)
            if complete_years:
                yearly_complete = yearly_complete.loc[yearly_complete.index.isin(complete_years)]

        display_cols = [
            "breach_count",
            "total_affected",
            "avg_affected",
            "median_affected",
            "unique_entities",
            "unique_states",
            "yoy_count_change",
            "yoy_affected_change",
        ]
        yearly_display = yearly_complete[display_cols].copy()
        yearly_display = yearly_display.reset_index()

        # Make column headers human-readable
        yearly_display = yearly_display.rename(
            columns={
                "year": "Year",
                "breach_count": "Breaches",
                "total_affected": "Total Affected",
                "avg_affected": "Avg Affected",
                "median_affected": "Median Affected",
                "unique_entities": "Unique Entities",
                "unique_states": "States Affected",
                "yoy_count_change": "YoY Breach Change (%)",
                "yoy_affected_change": "YoY Affected Change (%)",
            }
        )
        # Ensure Year is a simple string without thousands separators
        yearly_display["Year"] = yearly_display["Year"].astype(str)

        st.dataframe(
            yearly_display.style.format(
                {
                    "Breaches": "{:,.0f}",
                    "Total Affected": "{:,.0f}",
                    "Avg Affected": "{:,.0f}",
                    "Median Affected": "{:,.0f}",
                    "Unique Entities": "{:,.0f}",
                    "States Affected": "{:,.0f}",
                    "YoY Breach Change (%)": "{:+.1f}%",
                    "YoY Affected Change (%)": "{:+.1f}%",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

        yoy_df = (
            yearly_complete.reset_index()[["year", "yoy_count_change"]]
            .dropna()
        )
        fig_yoy = create_yoy_growth(yoy_df)
        st.plotly_chart(fig_yoy, use_container_width=True)

    st.markdown("---")
    st.subheader("Evolution of Breach Types")
    st.caption(
        "External cyberattacks (Hacking/IT Incidents) consistently account for over 80% of all compromises, "
        "vastly outnumbering physical theft or internal leaks."
    )

    # Prepare data for stacked area chart
    breach_type_df = filtered_df.copy()
    breach_type_df["date"] = breach_type_df["breach_date"]
    fig_breach_type = create_breach_type_area(breach_type_df)
    st.plotly_chart(fig_breach_type, use_container_width=True)

    render_footer()


if __name__ == "__main__":
    main()

