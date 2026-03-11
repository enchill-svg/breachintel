from __future__ import annotations

from typing import Any, Dict, List, Tuple

import pandas as pd
import streamlit as st


def configure_time_filters(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Render top-of-page time controls and return a date-filtered DataFrame plus metadata.

    - Canonical date field: 'breach_date'
    - Presets: Last 12 months, Last 5 years, All time, Custom
    """
    data = df.copy()
    if "breach_date" not in data.columns:
        return data, {}

    data["breach_date"] = pd.to_datetime(data["breach_date"], errors="coerce")
    data = data.dropna(subset=["breach_date"])
    if data.empty:
        return data, {}

    min_ts = data["breach_date"].min()
    max_ts = data["breach_date"].max()
    min_date = min_ts.date()
    max_date = max_ts.date()

    # Default window: last 5 years or full history if shorter
    default_start_ts = max_ts - pd.DateOffset(years=5)
    if default_start_ts.date() < min_date:
        default_start_date = min_date
    else:
        default_start_date = default_start_ts.date()

    preset = st.selectbox(
        "Date range",
        options=["Last 12 months", "Last 5 years", "All time", "Custom"],
        index=1,
        key="date_range_preset",
    )

    start_date: Any
    end_date: Any

    if preset == "Last 12 months":
        start_ts = max_ts - pd.DateOffset(years=1)
        start_date = max(start_ts.date(), min_date)
        end_date = max_date
    elif preset == "Last 5 years":
        start_date = default_start_date
        end_date = max_date
    elif preset == "All time":
        start_date = min_date
        end_date = max_date
    else:
        # Custom range with explicit FROM / TO pickers
        start_date, end_date = st.date_input(
            "From / To",
            value=(default_start_date, max_date),
            min_value=min_date,
            max_value=max_date,
            key="date_range_custom",
        )
        # Guard against inverted ranges
        if start_date > end_date:
            start_date, end_date = end_date, start_date

    mask = (data["breach_date"].dt.date >= start_date) & (
        data["breach_date"].dt.date <= end_date
    )
    filtered = data.loc[mask].copy()

    meta: Dict[str, Any] = {
        "start_date": start_date,
        "end_date": end_date,
        "preset": preset,
        "total_incidents": int(len(filtered)),
    }
    return filtered, meta


def _reset_sidebar_filters() -> None:
    """
    Clear sidebar filter widgets back to their defaults.
    """
    for key in ["Breach type", "Entity type", "State"]:
        if key in st.session_state:
            del st.session_state[key]
    # No explicit st.rerun() here: the button click that invokes this callback
    # already triggers a script rerun, so clearing the keys is sufficient.


def render_sidebar_filters(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    """
    Render grouped sidebar filters (breach type, entity type, geography) and return:
      - filtered DataFrame
      - filter state metadata for active filter summaries
    """
    st.sidebar.header("Filters")

    # Time section (top-level time is handled in the main layout)
    st.sidebar.markdown("### Time")
    st.sidebar.caption("Use the date controls at the top of the page.")

    data = df.copy()

    selected_breach_types: List[str] = []
    all_breach_types: List[str] = []
    selected_entity_types: List[str] = []
    all_entity_types: List[str] = []
    selected_states: List[str] = []
    all_states: List[str] = []

    # Breach type section
    st.sidebar.markdown("### Breach type")
    if "breach_type" in data.columns:
        all_breach_types = sorted(
            [bt for bt in data["breach_type"].dropna().unique().tolist()]
        )
        selected_breach_types = st.sidebar.multiselect(
            "",
            options=all_breach_types,
            default=all_breach_types,
            key="Breach type",
            label_visibility="collapsed",
        )
        if selected_breach_types:
            data = data[data["breach_type"].isin(selected_breach_types)]

    # Entity type section
    st.sidebar.markdown("### Entity type")
    if "entity_type" in data.columns:
        all_entity_types = sorted(
            [et for et in data["entity_type"].dropna().unique().tolist()]
        )
        selected_entity_types = st.sidebar.multiselect(
            "",
            options=all_entity_types,
            default=all_entity_types,
            key="Entity type",
            label_visibility="collapsed",
        )
        if selected_entity_types:
            data = data[data["entity_type"].isin(selected_entity_types)]

    # Geography section
    st.sidebar.markdown("### Geography")
    if "state" in data.columns:
        all_states = sorted([s for s in data["state"].dropna().unique().tolist()])
        selected_states = st.sidebar.multiselect(
            "State",
            options=all_states,
            default=all_states,
            key="State",
        )
        if selected_states:
            data = data[data["state"].isin(selected_states)]

    st.sidebar.button("Reset filters", on_click=_reset_sidebar_filters)

    filter_state: Dict[str, List[str]] = {
        "selected_breach_types": selected_breach_types or all_breach_types,
        "all_breach_types": all_breach_types,
        "selected_entity_types": selected_entity_types or all_entity_types,
        "all_entity_types": all_entity_types,
        "selected_states": selected_states or all_states,
        "all_states": all_states,
    }

    return data, filter_state


def render_active_filter_bar(
    time_meta: Dict[str, Any] | None,
    filter_state: Dict[str, List[str]] | None,
) -> None:
    """
    Display a compact summary of active filters as "chips" at the top of the page.
    """
    if not time_meta and not filter_state:
        return

    chips: List[str] = []

    if time_meta:
        start = time_meta.get("start_date")
        end = time_meta.get("end_date")
        if start and end:
            chips.append(f"<span class='bi-chip'>Date: {start} – {end}</span>")

    if filter_state:
        bt_selected = filter_state.get("selected_breach_types") or []
        bt_all = filter_state.get("all_breach_types") or []
        # Only show a chip when the selection is narrower than "All"
        if bt_all and 0 < len(bt_selected) < len(bt_all):
            shown = sorted(bt_selected)[:3]
            extra = len(bt_selected) - len(shown)
            label = ", ".join(shown) + (f" (+{extra})" if extra > 0 else "")
            chips.append(f"<span class='bi-chip'>Type: {label}</span>")

        et_selected = filter_state.get("selected_entity_types") or []
        et_all = filter_state.get("all_entity_types") or []
        if et_all and 0 < len(et_selected) < len(et_all):
            shown = sorted(et_selected)[:3]
            extra = len(et_selected) - len(shown)
            label = ", ".join(shown) + (f" (+{extra})" if extra > 0 else "")
            chips.append(f"<span class='bi-chip'>Entity: {label}</span>")

        st_selected = filter_state.get("selected_states") or []
        st_all = filter_state.get("all_states") or []
        if st_all and 0 < len(st_selected) < len(st_all):
            shown = sorted(st_selected)[:3]
            extra = len(st_selected) - len(shown)
            label = ", ".join(shown) + (f" (+{extra})" if extra > 0 else "")
            chips.append(f"<span class='bi-chip'>State: {label}</span>")

    if not chips:
        return

    chips_joined = "".join(chips)
    chips_html = f"""
    <style>
    .bi-chip-container {{
        display: flex;
        flex-wrap: wrap;
        gap: 0.4rem;
        margin-bottom: 0.75rem;
        font-size: 0.8rem;
    }}
    .bi-chip {{
        padding: 0.15rem 0.5rem;
        border-radius: 999px;
        border: 1px solid rgba(148, 163, 184, 0.6);
        background-color: rgba(15, 23, 42, 0.7);
        color: #E5E7EB;
    }}
    </style>
    <div class="bi-chip-container">
        {chips_joined}
    </div>
    """
    st.markdown(chips_html, unsafe_allow_html=True)
