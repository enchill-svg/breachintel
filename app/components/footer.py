from __future__ import annotations

import streamlit as st


def render_footer() -> None:
    """Render a consistent footer."""
    st.markdown(
        """
        <hr style="margin-top:2rem; border-color: #1F2937;" />
        <div style="text-align:center; font-size:0.8rem; color:#6B7280; padding:0.75rem 0 0.5rem 0;">
            Built by Yewku Enchill-Yawson | Data source: HHS OCR Breach Portal | MIT License
        </div>
        """,
        unsafe_allow_html=True,
    )

