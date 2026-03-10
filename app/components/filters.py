from __future__ import annotations

import streamlit as st
import pandas as pd


def render_sidebar_filters(df: pd.DataFrame) -> pd.DataFrame:
    """
    Render standard sidebar filters and return a filtered DataFrame.
    """
    st.sidebar.header("Filters")

    # Ensure breach_date is datetime for year extraction
    data = df.copy()
    if "breach_date" in data.columns:
        data["breach_date"] = pd.to_datetime(data["breach_date"], errors="coerce")
        data["year"] = data["breach_date"].dt.year

    # Year range slider
    if "year" in data.columns and data["year"].notna().any():
        min_year = int(data["year"].min())
        max_year = int(data["year"].max())
        start_year, end_year = st.sidebar.slider(
            "Year range",
            min_value=min_year,
            max_value=max_year,
            value=(min_year, max_year),
        )
        data = data[(data["year"] >= start_year) & (data["year"] <= end_year)]

    # Breach type multiselect
    if "breach_type" in data.columns:
        breach_types = sorted(
            [bt for bt in data["breach_type"].dropna().unique().tolist()]
        )
        selected_breach_types = st.sidebar.multiselect(
            "Breach type",
            options=breach_types,
            default=breach_types,
        )
        if selected_breach_types:
            data = data[data["breach_type"].isin(selected_breach_types)]

    # Entity type multiselect
    if "entity_type" in data.columns:
        entity_types = sorted(
            [et for et in data["entity_type"].dropna().unique().tolist()]
        )
        selected_entity_types = st.sidebar.multiselect(
            "Entity type",
            options=entity_types,
            default=entity_types,
        )
        if selected_entity_types:
            data = data[data["entity_type"].isin(selected_entity_types)]

    # State multiselect
    if "state" in data.columns:
        states = sorted([s for s in data["state"].dropna().unique().tolist()])
        selected_states = st.sidebar.multiselect(
            "State",
            options=states,
            default=states,
        )
        if selected_states:
            data = data[data["state"].isin(selected_states)]

    return data

