from __future__ import annotations

from typing import Optional

import streamlit as st

from breachintel.utils.constants import COLORS


def render_kpi_card(
    title: str,
    value: str,
    delta: Optional[str] = None,
    delta_label: Optional[str] = None,
    color: Optional[str] = None,
) -> None:
    """Render a styled KPI card using HTML/CSS."""
    border_color = COLORS.get(color, COLORS["primary"]) if color else COLORS["primary"]

    delta_html = ""
    if delta is not None:
        label_part = (
            f"<span style='opacity:0.8;'>{delta_label}</span> "
            if delta_label
            else ""
        )
        delta_html = (
            f"<div style=\"font-size:0.8rem; color:{COLORS['text_secondary']}; "
            f"margin-top:0.25rem;\">{label_part}<span>{delta}</span></div>"
        )

    card_html = (
        f"<div style=\""
        f"background-color:{COLORS['bg_card']};"
        f"border:1px solid {border_color}33;"
        f"border-radius:0.75rem;"
        f"padding:0.9rem 1rem;"
        f"box-shadow:0 10px 20px rgba(15,23,42,0.45);"
        f"\">"
        f"<div style=\"font-size:0.8rem; text-transform:uppercase; "
        f"letter-spacing:0.08em; color:{COLORS['text_secondary']};\">"
        f"{title}</div>"
        f"<div style=\"font-size:1.4rem; font-weight:600; "
        f"color:{COLORS['text_primary']}; margin-top:0.25rem;\">"
        f"{value}</div>"
        f"{delta_html}"
        f"</div>"
    )
    st.markdown(card_html, unsafe_allow_html=True)


def render_severity_badge(severity: str) -> str:
    """Return an HTML badge styled by severity level."""
    severity = (severity or "").strip().title()
    color_map = {
        "Low": COLORS["success"],
        "Medium": COLORS["warning"],
        "High": COLORS["danger"],
        "Critical": "#DC2626",
    }
    bg = color_map.get(severity, COLORS["info"])

    return (
        f"<span style=\""
        f"display:inline-flex;"
        f"align-items:center;"
        f"padding:0.15rem 0.5rem;"
        f"border-radius:999px;"
        f"font-size:0.75rem;"
        f"font-weight:500;"
        f"background-color:{bg}33;"
        f"color:{bg};"
        f"text-transform:uppercase;"
        f"letter-spacing:0.06em;"
        f"\">"
        f"{severity or 'Unknown'}"
        f"</span>"
    )

