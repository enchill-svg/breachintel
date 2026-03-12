from __future__ import annotations

from typing import Any, Dict, Optional

import pandas as pd
import streamlit as st

from breachintel.utils.constants import COLORS, STATE_ABBREVIATIONS

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
    show_frame: bool = True,
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
    state_raw = _coerce_str(row.get("state"))
    # For non-technical users, expand state abbreviations where possible and
    # display them in title case (e.g., "Minnesota" instead of "MN" or "MINNESOTA").
    # STATE_ABBREVIATIONS maps many variants to USPS codes; we invert to a simple
    # code -> title-cased full name map when possible.
    code_to_name: Dict[str, str] = {}
    for full_or_code, code in STATE_ABBREVIATIONS.items():
        if len(full_or_code) > 2:  # treat this as a full state/territory name
            code_to_name[code] = full_or_code.title()
    # Try to resolve by treating the raw value as a code; fall back to title-cased raw.
    candidate = code_to_name.get(state_raw.upper())
    state = candidate if candidate is not None else state_raw.title()

    individuals = row.get("individuals_affected")
    individuals_display = (
        f"{int(individuals):,}"
        if isinstance(individuals, (int, float)) and not pd.isna(individuals)
        else "Unknown"
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

    # Use Streamlit layout primitives instead of a large raw HTML string.
    with st.container(border=False):
        if show_frame:
            # Opening div and title in one markdown so no empty bordered box appears above the title.
            st.markdown(
                f"""
<div style="
  background-color:{COLORS['bg_card']};
  border-radius:0.85rem;
  border:1px solid rgba(148,163,184,0.4);
  padding:0.85rem 1.1rem 0.75rem 1.1rem;
  margin-top:0.4rem;
">
<div style='font-size:0.78rem; text-transform:uppercase; letter-spacing:0.08em; color:#CBD5F5;'>{title}</div>
""",
                unsafe_allow_html=True,
            )

        col_left, col_right = st.columns(2)

        with col_left:
            st.markdown(
                f"<div style='font-size:0.75rem; text-transform:uppercase; "
                f"letter-spacing:0.08em; color:#CBD5F5; margin-top:0.35rem;'>Entity</div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<div style='font-weight:600; color:{COLORS['text_primary']};'>"
                f"{entity_name}</div>",
                unsafe_allow_html=True,
            )
            # Severity badge anchored next to entity name block
            st.markdown(
                f"<div style='margin-top:0.25rem;'>{severity_badge}</div>",
                unsafe_allow_html=True,
            )
            # Map business associate terminology to a plainer label
            entity_type_display = (
                "Third-Party Vendor" if entity_type.lower() == "business associate" else entity_type
            )
            st.markdown(
                f"<div style='color:{COLORS['text_secondary']}; margin-top:0.1rem;'>"
                f"{entity_type_display} · {state}</div>",
                unsafe_allow_html=True,
            )

            st.markdown(
                f"<div style='font-size:0.75rem; text-transform:uppercase; "
                f"letter-spacing:0.08em; color:#CBD5F5; "
                f"margin-top:0.75rem;'>Impact</div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<div style='font-weight:600; color:{COLORS['text_primary']};'>"
                f"{individuals_display} individuals</div>",
                unsafe_allow_html=True,
            )
            # Approximate share of the U.S. population for context (e.g. 500k -> 0.15%)
            try:
                n_individuals = float(individuals) if individuals is not None else 0.0
                pct_pop = (n_individuals / 330_000_000) * 100
                pct_text = f"Approximately {pct_pop:.2f}% of the U.S. population"
            except (TypeError, ValueError, ZeroDivisionError):
                pct_text = ""
            if pct_text:
                st.markdown(
                    f"<div style='color:{COLORS['text_secondary']}; margin-top:0.1rem; font-size:0.8rem;'>"
                    f"{pct_text}</div>",
                    unsafe_allow_html=True,
                )
            st.markdown(
                f"<div style='color:{COLORS['text_secondary']}; margin-top:0.1rem;'>"
                f"Involved a third-party vendor: {ba_display}</div>",
                unsafe_allow_html=True,
            )

        with col_right:
            st.markdown(
                f"<div style='font-size:0.75rem; text-transform:uppercase; "
                f"letter-spacing:0.08em; color:#CBD5F5;'>Timeline</div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<div style='color:{COLORS['text_primary']};'>"
                f"Breach Submission Date: <strong>{breach_date_display}</strong></div>",
                unsafe_allow_html=True,
            )

            st.markdown(
                f"<div style='font-size:0.75rem; text-transform:uppercase; "
                f"letter-spacing:0.08em; color:#CBD5F5; "
                f"margin-top:0.75rem;'>Classification</div>",
                unsafe_allow_html=True,
            )
            # Plain-language labels for non-technical users
            st.markdown(
                f"<div style='color:{COLORS['text_primary']};'>"
                f"Cause: <strong>{'Cyberattack' if breach_type == 'Hacking/IT Incident' else breach_type}</strong></div>",
                unsafe_allow_html=True,
            )
            # Only show attack vector when it conveys useful information
            if attack_category_display.lower() != "unknown":
                st.markdown(
                    f"<div style='color:{COLORS['text_primary']}; margin-top:0.1rem;'>"
                    f"Attack vector: <strong>{attack_category_display}</strong></div>",
                    unsafe_allow_html=True,
                )
            st.markdown(
                f"<div style='color:{COLORS['text_secondary']}; margin-top:0.1rem;'>"
                f"Compromised System: {breach_location}</div>",
                unsafe_allow_html=True,
            )

        if show_frame:
            st.markdown("</div>", unsafe_allow_html=True)

