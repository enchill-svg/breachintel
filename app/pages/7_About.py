from __future__ import annotations

import streamlit as st

from app.components.footer import render_footer


st.set_page_config(
    page_title="BreachIntel — About",
    page_icon="ℹ️",
    layout="wide",
    initial_sidebar_state="expanded",
)


def main() -> None:
    st.title("ℹ️ About BreachIntel")

    st.markdown(
        """
        BreachIntel is an analytical platform focused on **healthcare data breaches**. It
        brings together regulatory breach disclosures, statistical analysis, and machine
        learning to help security teams, compliance officers, and researchers understand
        how real-world incidents are evolving over time.

        The dashboard surfaces trends in breach volume, geography, attack vectors, and
        entity risk profiles, and includes predictive components such as a breach
        severity estimator and forward-looking breach forecast. The goal is not to
        replace human judgment, but to provide **decision-quality context** in a compact,
        explorable interface.
        """
    )

    st.header("Data Source")
    st.markdown(
        """
        The underlying dataset is derived from the **U.S. Department of Health and Human
        Services (HHS) Office for Civil Rights (OCR) Breach Portal**, sometimes referred
        to as the *“Wall of Shame”*. This portal tracks breaches of unsecured protected
        health information (PHI) reported under the HIPAA Breach Notification Rule.

        The dataset typically includes:

        - Covered entity name and type
        - State and approximate incident date
        - Breach type (e.g., Hacking/IT Incident, Unauthorized Access/Disclosure)
        - Location of breached information (e.g., Network Server, Email, Paper/Films)
        - Number of individuals affected

        The date range depends on the locally downloaded snapshot, but usually covers
        **multiple years of historical incidents**, updated as new breaches are posted.
        """
    )

    st.header("Methodology")
    st.markdown(
        """
        BreachIntel combines several analytical modules:

        - **TrendAnalyzer**: Computes monthly and yearly breach trends, moving averages,
          and inflection points to highlight accelerating or decelerating risk.
        - **GeographicAnalyzer**: Normalizes breach activity by population, producing
          per-capita risk views and state-level summaries.
        - **AttackVectorAnalyzer**: Focuses on breach types, locations, and severity
          distributions, including a breach-type × severity matrix.
        - **EntityProfiler**: Builds entity-level risk profiles, repeat-offender lists,
          and entity-type trends over time.
        - **SeverityModel**: A Random Forest classifier trained to predict breach severity
          tiers from structured features.
        - **BreachForecaster**: A Prophet-based time-series forecaster that projects
          future monthly breach counts with uncertainty intervals.
        """
    )

    st.header("Limitations")
    st.markdown(
        """
        When interpreting the charts and model outputs, keep these constraints in mind:

        - **Thresholded reporting**: Only breaches affecting **500+ individuals** are
          included in the public HHS OCR portal; smaller incidents are absent.
        - **Scope**: The data only covers **U.S. HIPAA-covered entities** and their
          business associates.
        - **Model generalization**: Machine learning models are trained on **historical
          patterns** and may not predict **novel attack types** or future regulatory
          changes.
        - **Self-reported data**: Breach descriptions and categorizations are
          **self-reported** and may be incomplete, inconsistent, or delayed.
        - **Forecast assumptions**: Time-series forecasts implicitly assume that
          historical structure and seasonality **continue into the future** and do not
          account for black-swan events.
        """
    )

    st.header("Tech Stack")
    st.markdown(
        """
        - **Backend & analysis**: Python, pandas, NumPy
        - **Visualization**: Plotly, Folium, Streamlit
        - **Machine learning**: scikit-learn (Random Forests), SHAP for explainability
        - **Time series**: Prophet (additive time-series forecasting)
        - **Configuration & logging**: Pydantic settings, structured logging utilities
        """
    )

    st.header("Author & License")
    st.markdown(
        """
        **Author**: Your Name — *Your role / credentials here (e.g., Security Engineer,
        Data Scientist, Healthcare Privacy Researcher)*.

        **License**: MIT. You are free to use, modify, and redistribute this project
        under the terms of the MIT License.

        **Source code**: The BreachIntel project is available on GitHub:
        `[GitHub repository URL here]`.
        """
    )

    render_footer()


if __name__ == "__main__":
    main()

