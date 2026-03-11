from __future__ import annotations

from typing import Any, Dict, Optional

import pandas as pd
import streamlit as st

from breachintel.utils.constants import COLORS

from app.components.metrics import render_severity_badge


def _coerce_str(value: Any, default: str = "Unknown") -> str:
    if value is None:
        return default
    if isinstance(value, float) and pd.isna(value):
        return default
    text = str(value).strip()
    return text or default


def _infer_severity_from_count(individuals_affected: Any) -> str:
    try:
        n = float(individuals_affected)
    except (TypeError, ValueError):
        return "Unknown"

    if n < 1_000:
        return "Low"
    if n < 100_000:
        return "Medium"
    if n < 1_000_000:
        return "High"
    return "Critical"


def render_breach_detail_card(
    record: pd.Series | Dict[str, Any],
    predicted_severity: Optional[str] = None,
    attack_category: Optional[str] = None,
    title: str = "Breach Details",
) -> None:
    """
    Render a reusable Breach Detail Card from a single breach record.
    """
    if isinstance(record, dict):
        row = pd.Series(record)
    else:
        row = record

    entity_name = _coerce_str(row.get("entity_name"), default="Unknown entity")
    entity_type = _coerce_str(row.get("entity_type"))
    state = _coerce_str(row.get("state"))

    individuals = row.get("individuals_affected")
    individuals_display = (
        f"{int(individuals):,}" if isinstance(individuals, (int, float)) and not pd.isna(individuals) else "Unknown"
    )

    ba_flag = _coerce_str(row.get("business_associate"))
    if ba_flag.lower() in {"yes", "y", "true", "1"}:
        ba_display = "Yes"
    elif ba_flag.lower() in {"no", "n", "false", "0"}:
        ba_display = "No"
    else:
        ba_display = "Unknown"

    # Timeline
    breach_date = row.get("breach_date")
    submitted_date = row.get("date_submitted") or row.get("submission_date")

    def _fmt_dt(value: Any) -> str:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return "Unknown"
        try:
            dt = pd.to_datetime(value, errors="coerce")
            if pd.isna(dt):
                return "Unknown"
            return dt.strftime("%Y-%m-%d")
        except Exception:
            return "Unknown"

    breach_date_display = _fmt_dt(breach_date)
    submitted_date_display = _fmt_dt(submitted_date)

    # Categories
    breach_type = _coerce_str(row.get("breach_type"))
    breach_location = _coerce_str(row.get("breach_location"))

    if attack_category is None:
        attack_category = row.get("attack_category") or row.get("nlp_attack_category")
    attack_category_display = _coerce_str(attack_category)

    # Severity: prefer explicit severity, then predicted severity, then heuristic
    base_severity = row.get("severity")
    severity_label = None
    if isinstance(base_severity, str) and base_severity.strip():
        severity_label = base_severity
    elif isinstance(predicted_severity, str) and predicted_severity.strip():
        severity_label = predicted_severity
    else:
        severity_label = _infer_severity_from_count(individuals)

    severity_badge = render_severity_badge(severity_label)

    # Use Streamlit layout primitives instead of a large raw HTML string
    with st.container():
        st.markdown(
            f"""
<div style="
  background-color:{COLORS['bg_card']};
  border-radius:0.9rem;
  border:1px solid rgba(148,163,184,0.5);
  padding:1rem 1.2rem;
  margin-top:0.5rem;
">
""",
            unsafe_allow_html=True,
        )

        header_cols = st.columns([3, 1])
        with header_cols[0]:
            st.markdown(
                f"<div style='font-size:0.9rem; text-transform:uppercase; "
                f"letter-spacing:0.1em; color:{COLORS['text_secondary']};'>{title}</div>",
                unsafe_allow_html=True,
            )
        with header_cols[1]:
            st.markdown(severity_badge, unsafe_allow_html=True)

        col_left, col_right = st.columns(2)

        with col_left:
            st.markdown(
                f"<div style='font-size:0.75rem; text-transform:uppercase; "
                f"letter-spacing:0.08em; color:{COLORS['text_secondary']};'>Entity</div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<div style='font-weight:600; color:{COLORS['text_primary']};'>"
                f"{entity_name}</div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<div style='color:{COLORS['text_secondary']}; margin-top:0.1rem;'>"
                f"{entity_type} · {state}</div>",
                unsafe_allow_html=True,
            )

            st.markdown(
                f"<div style='font-size:0.75rem; text-transform:uppercase; "
                f"letter-spacing:0.08em; color:{COLORS['text_secondary']}; "
                f"margin-top:0.75rem;'>Impact</div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<div style='font-weight:600; color:{COLORS['text_primary']};'>"
                f"{individuals_display} individuals</div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<div style='color:{COLORS['text_secondary']}; margin-top:0.1rem;'>"
                f"Business associate present: {ba_display}</div>",
                unsafe_allow_html=True,
            )

        with col_right:
            st.markdown(
                f"<div style='font-size:0.75rem; text-transform:uppercase; "
                f"letter-spacing:0.08em; color:{COLORS['text_secondary']};'>Timeline</div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<div style='color:{COLORS['text_primary']};'>"
                f"Breach Submission Date: <strong>{breach_date_display}</strong></div>",
                unsafe_allow_html=True,
            )

            st.markdown(
                f"<div style='font-size:0.75rem; text-transform:uppercase; "
                f"letter-spacing:0.08em; color:{COLORS['text_secondary']}; "
                f"margin-top:0.75rem;'>Classification</div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<div style='color:{COLORS['text_primary']};'>"
                f"Breach type: <strong>{breach_type}</strong></div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<div style='color:{COLORS['text_primary']}; margin-top:0.1rem;'>"
                f"Attack vector: <strong>{attack_category_display}</strong></div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<div style='color:{COLORS['text_secondary']}; margin-top:0.1rem;'>"
                f"Location: {breach_location}</div>",
                unsafe_allow_html=True,
            )

        st.markdown("</div>", unsafe_allow_html=True)

