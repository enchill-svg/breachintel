from __future__ import annotations

from typing import Dict

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from breachintel.utils.constants import COLORS


def apply_theme(fig: go.Figure) -> go.Figure:
    """Apply the shared dark theme to a Plotly figure."""
    fig.update_layout(
        paper_bgcolor=COLORS["bg_dark"],
        plot_bgcolor=COLORS["bg_dark"],
        font=dict(
            color=COLORS["text_primary"],
            family="Inter, system-ui, -apple-system, BlinkMacSystemFont, sans-serif",
            size=13,
        ),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
        ),
        margin=dict(l=60, r=30, t=50, b=50),
    )

    fig.update_xaxes(gridcolor=COLORS["grid"])
    fig.update_yaxes(gridcolor=COLORS["grid"])

    return fig


def create_overview_chart(df: pd.DataFrame) -> go.Figure:
    """Monthly breach count with 12‑month moving average. Expects row-level df with a date column."""
    data = df.copy()
    cols = [str(c) for c in data.columns]
    date_col = None
    for name in ("breach_date", "date", "Breach Submission Date"):
        if name in cols:
            date_col = name
            break
    if date_col is None:
        for c in cols:
            if "date" in c.lower():
                date_col = c
                break
    if date_col is None:
        raise KeyError(f"DataFrame must contain a date column (e.g. 'breach_date' or 'date'); got {cols}")
    # Work with a single canonical column so set_index never sees a missing name
    data["breach_date"] = pd.to_datetime(data[date_col], errors="coerce")
    data = data.dropna(subset=["breach_date"])

    count_col = "entity_name" if "entity_name" in data.columns else data.columns[0]
    monthly = (
        data.set_index("breach_date")
        .resample("ME")
        .agg(count=(count_col, "size"))
        .reset_index()
    )
    monthly["ma12"] = monthly["count"].rolling(12, min_periods=6).mean()

    # Ensure counts are integers
    monthly["count"] = monthly["count"].astype(int)

    fig = go.Figure()

    fig.add_bar(
        x=monthly["breach_date"],
        y=monthly["count"],
        name="Monthly Breaches",
        marker_color="#00E6B8",
        opacity=0.7,
    )

    fig.add_trace(
        go.Scatter(
            x=monthly["breach_date"],
            y=monthly["ma12"],
            mode="lines",
            name="12‑Month Moving Average",
            line=dict(color=COLORS["secondary"], width=3),
        )
    )

    fig.update_layout(
        title={
            "text": "Healthcare Data Breaches Over Time",
            "x": 0.5,
            "xanchor": "center",
        },
        hovermode="x unified",
        # Reduce top margin to avoid excessive empty space above the title
        margin=dict(t=30, l=60, r=30, b=50),
    )

    # Clean, readable hover tooltips for non-technical users
    fig.update_traces(
        selector=dict(name="Monthly Breaches"),
        hovertemplate="Month: %{x|%b %Y}<br>Monthly breaches: %{y:.0f}<extra></extra>",
    )
    fig.update_traces(
        selector=dict(name="12‑Month Moving Average"),
        hovertemplate="Month: %{x|%b %Y}<br>12‑month average: %{y:.1f}<extra></extra>",
    )
    fig.update_yaxes(rangemode="tozero")

    return apply_theme(fig)


def create_breach_type_area(df: pd.DataFrame) -> go.Figure:
    """100% stacked bar chart of breach types over years (x=year, y=percentage)."""
    data = df.copy()
    if "year" not in data.columns and "breach_date" in data.columns:
        data["year"] = pd.to_datetime(data["breach_date"], errors="coerce").dt.year
    elif "year" not in data.columns and "date" in data.columns:
        data["year"] = pd.to_datetime(data["date"], errors="coerce").dt.year
    if "breach_type" not in data.columns:
        data["breach_type"] = "Unknown"

    evolution = (
        data.dropna(subset=["year"])
        .groupby(["year", "breach_type"])
        .size()
        .reset_index(name="count")
    )

    # Convert to 100% stacked bar by normalizing counts within each year
    evolution["year_total"] = evolution.groupby("year")["count"].transform("sum")
    evolution["percentage"] = evolution["count"] / evolution["year_total"]

    fig = px.bar(
        evolution,
        x="year",
        y="percentage",
        color="breach_type",
        title=None,
    )
    fig.update_layout(
        barmode="stack",
        colorway=list(COLORS.values()),
        xaxis_title="Year",
        yaxis_title="Percentage of Breaches",
        legend_title_text="Breach Type",
        yaxis=dict(tickformat=".0%", rangemode="tozero"),
    )
    return apply_theme(fig)


def create_severity_distribution(df: pd.DataFrame) -> go.Figure:
    """Donut chart of severity distribution."""
    severity_counts = df["severity"].value_counts().reindex(
        ["Low", "Medium", "High", "Critical"]
    )

    color_map: Dict[str, str] = {
        "Low": COLORS["success"],
        "Medium": COLORS["warning"],
        "High": COLORS["danger"],
        "Critical": "#DC2626",
    }

    labels = severity_counts.index.tolist()
    values = severity_counts.values.tolist()
    colors = [color_map.get(label, COLORS["info"]) for label in labels]

    fig = go.Figure(
        go.Pie(
            labels=labels,
            values=values,
            hole=0.5,
            marker=dict(colors=colors),
            textfont=dict(color="#FFFFFF"),
        )
    )

    fig.update_layout(title="Severity Distribution")

    return apply_theme(fig)


def create_forecast_chart(
    forecast_df: pd.DataFrame, actual_df: pd.DataFrame
) -> go.Figure:
    """Forecast chart with confidence interval and actuals."""
    # Normalize forecast dates so x-axis is consistently datetime
    forecast_df = forecast_df.copy()
    forecast_df["date"] = pd.to_datetime(forecast_df["date"], errors="coerce")
    forecast_df = forecast_df.dropna(subset=["date"])

    # Build actual monthly series: use count column (breach_count or y) per month
    actual = actual_df.copy()
    actual["date"] = pd.to_datetime(actual["date"], errors="coerce")
    actual = actual.dropna(subset=["date"])
    count_col = "breach_count" if "breach_count" in actual.columns else "y"
    if count_col not in actual.columns:
        other = [c for c in actual.columns if c != "date"]
        count_col = other[0] if other else None
    if count_col is None:
        actual_monthly = pd.DataFrame(
            {"date": pd.DatetimeIndex([]), "breach_count": []}
        ).set_index("date")
    else:
        actual_monthly = (
            actual.set_index("date")
            .resample("ME")
            .agg({count_col: "sum"})
            .rename(columns={count_col: "breach_count"})
        )
    # Ensure index is datetime and y is numeric for Plotly
    actual_monthly.index = pd.to_datetime(actual_monthly.index)
    actual_monthly["breach_count"] = actual_monthly["breach_count"].astype(float)

    fig = go.Figure()

    # Confidence interval: closed polygon using forecast date and lower/upper
    if {"date", "lower", "upper"}.issubset(forecast_df.columns):
        x = forecast_df["date"].tolist()
        y_upper = forecast_df["upper"].astype(float).tolist()
        y_lower = forecast_df["lower"].astype(float).tolist()
        fig.add_trace(
            go.Scatter(
                x=x + x[::-1],
                y=y_upper + y_lower[::-1],
                fill="toself",
                fillcolor="rgba(0, 212, 170, 0.15)",
                line=dict(color="rgba(0,0,0,0)"),
                hoverinfo="skip",
                showlegend=True,
                name="90% CI",
            )
        )

    # Forecast line: x=date, y=forecast
    forecast_col = "forecast"
    if forecast_col not in forecast_df.columns and "yhat" in forecast_df.columns:
        forecast_col = "yhat"
    if {"date", forecast_col}.issubset(forecast_df.columns):
        fig.add_trace(
            go.Scatter(
                x=forecast_df["date"],
                y=forecast_df[forecast_col].astype(float),
                mode="lines",
                name="Forecast",
                line=dict(color=COLORS["primary"], dash="dash"),
            )
        )

    # Actual line: only add if we have data; x=datetime index, y=breach_count
    if len(actual_monthly) > 0:
        fig.add_trace(
            go.Scatter(
                x=actual_monthly.index,
                y=actual_monthly["breach_count"].values,
                mode="lines",
                name="Actual",
                line=dict(color=COLORS["secondary"]),
            )
        )

    fig.update_layout(
        title="Breach Forecast (24 Months)",
        hovermode="x unified",
    )
    fig.update_xaxes(
        showticklabels=True,
        type="date",
        tickformat="%b %Y",
    )
    fig.update_yaxes(rangemode="tozero")

    return apply_theme(fig)


def create_entity_comparison(entity_summary: pd.DataFrame) -> go.Figure:
    """Horizontal bar chart comparing entity types by breach_count."""
    fig = px.bar(
        entity_summary,
        x="breach_count",
        y="entity_type",
        orientation="h",
        color="entity_type",
        color_discrete_sequence=list(COLORS.values()),
        title="Breaches by Entity Type",
    )

    return apply_theme(fig)


def create_yoy_growth(yearly: pd.DataFrame) -> go.Figure:
    """Year‑over‑year breach count change."""
    colors = [
        COLORS["danger"] if val >= 0 else COLORS["success"]
        for val in yearly["yoy_count_change"]
    ]

    fig = go.Figure(
        go.Bar(
            x=yearly["year"].astype(str),
            y=yearly["yoy_count_change"],
            marker_color=colors,
            name="YoY Change in Breach Count",
        )
    )

    fig.update_layout(
        title="Year‑over‑Year Breach Growth",
        xaxis_type="category",
        xaxis_title="Year",
        yaxis_title="YoY Change (%)",
    )

    return apply_theme(fig)

