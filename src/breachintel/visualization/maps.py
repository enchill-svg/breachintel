from __future__ import annotations

from typing import List, Tuple

import numpy as np
import folium
from folium.plugins import HeatMap

from breachintel.utils.constants import STATE_COORDS, COLORS


US_CENTER: Tuple[float, float] = (39.8283, -98.5795)


def _format_int(value) -> str:
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return "N/A"


def create_breach_heatmap(state_summary) -> folium.Map:
    """Create a national breach heatmap by state."""
    m = folium.Map(
        location=US_CENTER,
        zoom_start=4,
        tiles="CartoDB dark_matter",
    )

    heat_data: List[Tuple[float, float, float]] = []

    for _, row in state_summary.iterrows():
        state = row.get("state")
        breach_count = row.get("breach_count", 0) or 0

        coords = STATE_COORDS.get(state)
        if not coords or breach_count <= 0:
            continue

        weight = float(np.log1p(breach_count))
        heat_data.append((coords[0], coords[1], weight))

        total_affected = row.get("total_affected")
        per_100k = row.get("breaches_per_100k")

        tooltip_lines = [
            f"State: {state}",
            f"Breaches: {_format_int(breach_count)}",
        ]

        if total_affected is not None:
            tooltip_lines.append(f"Individuals affected: {_format_int(total_affected)}")

        if per_100k is not None:
            try:
                tooltip_lines.append(
                    f"Breaches per 100k: {float(per_100k):.2f}"
                )
            except (TypeError, ValueError):
                pass

        tooltip_text = "<br>".join(tooltip_lines)

        radius = max(3.0, min(20.0, float(breach_count) / 30.0))

        folium.CircleMarker(
            location=coords,
            radius=radius,
            color=COLORS["primary"],
            fill=True,
            fill_color=COLORS["primary"],
            fill_opacity=0.6,
            tooltip=folium.Tooltip(tooltip_text),
        ).add_to(m)

    if heat_data:
        HeatMap(
            heat_data,
            radius=30,
            blur=20,
            max_zoom=6,
            gradient={
                0.2: COLORS["success"],
                0.5: COLORS["warning"],
                0.8: COLORS["danger"],
                1.0: COLORS["danger"],
            },
        ).add_to(m)

    return m


def create_state_detail_map(df, state_code: str) -> folium.Map:
    """Create a detailed map for a single state."""
    coords = STATE_COORDS.get(state_code, US_CENTER)

    m = folium.Map(
        location=coords,
        zoom_start=7,
        tiles="CartoDB dark_matter",
    )

    state_df = df[df.get("state") == state_code] if hasattr(df, "get") else df

    # Prefer point-level markers when lat/lon are available
    if "latitude" in state_df.columns and "longitude" in state_df.columns:
        for _, row in state_df.iterrows():
            lat = row.get("latitude")
            lon = row.get("longitude")
            if lat is None or lon is None:
                continue

            breach_date = row.get("date")
            breach_type = row.get("breach_type")
            individuals = row.get("individuals_affected")

            lines = []
            if breach_date is not None:
                lines.append(f"Date: {breach_date}")
            if breach_type is not None:
                lines.append(f"Type: {breach_type}")
            if individuals is not None:
                lines.append(f"Individuals affected: {_format_int(individuals)}")

            popup_text = "<br>".join(lines) if lines else f"State: {state_code}"

            folium.CircleMarker(
                location=(lat, lon),
                radius=4,
                color=COLORS["secondary"],
                fill=True,
                fill_color=COLORS["secondary"],
                fill_opacity=0.7,
                popup=folium.Popup(popup_text, max_width=300),
            ).add_to(m)
    else:
        # Fallback to a single state-level marker
        breach_count = getattr(
            state_df["breach_count"], "sum", lambda: None
        )() if "breach_count" in state_df.columns else None
        total_affected = getattr(
            state_df["individuals_affected"], "sum", lambda: None
        )() if "individuals_affected" in state_df.columns else None

        lines = [f"State: {state_code}"]
        if breach_count is not None:
            lines.append(f"Breaches: {_format_int(breach_count)}")
        if total_affected is not None:
            lines.append(f"Individuals affected: {_format_int(total_affected)}")

        popup_text = "<br>".join(lines)

        folium.CircleMarker(
            location=coords,
            radius=6,
            color=COLORS["secondary"],
            fill=True,
            fill_color=COLORS["secondary"],
            fill_opacity=0.7,
            popup=folium.Popup(popup_text, max_width=300),
        ).add_to(m)

    return m

