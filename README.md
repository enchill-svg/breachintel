# 🛡️ BreachIntel

**Healthcare Cybersecurity Breach Intelligence Platform**

> Analyzing 15+ years of U.S. healthcare data breaches to predict severity, classify attack vectors, and forecast future trends.

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![CI](https://img.shields.io/badge/CI-passing-brightgreen.svg)
![Code style](https://img.shields.io/badge/code%20style-ruff-000000.svg)

[Documentation](docs/)

---

## Key Features

- **Trend Analysis**: Monthly and yearly breach trends with moving averages, inflection point detection, and YoY change.
- **Geographic Intelligence**: State-level and per-capita breach rates with interactive Folium heatmaps and state detail views.
- **Attack Vector Analysis**: Breach-type evolution, breach-location breakdown, and severity matrix (type × severity).
- **ML Severity Prediction with SHAP**: Random Forest classifier that estimates breach severity and explains predictions via SHAP.
- **NLP Attack Classification**: TF–IDF + Random Forest model that classifies free-text breach descriptions into attack categories.
- **24-Month Forecast**: Prophet-based monthly breach forecasts with confidence intervals and summary metrics.

---

## Quick Start

1. **Clone the repository**

   ```bash
   git clone https://github.com/enchill-svg/breachintel.git
   cd breachintel
   ```

2. **Create and activate a virtual environment**

   ```bash
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # macOS / Linux
   # source .venv/bin/activate
   ```

3. **Install dependencies**

   ```bash
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   python -m pip install -r requirements-dev.txt
   python -m pip install -e .
   ```

4. **Run the full data pipeline (download, clean, features)**

   ```bash
   python scripts/download_data.py
   ```

5. **Train models**

   ```bash
   # Create tabular features and train all models
   python scripts/train_models.py
   ```

6. **Run the Streamlit dashboard**

   ```bash
   streamlit run app/Home.py
   ```

7. **(Optional) Run tests + coverage**

   ```bash
   python -m pytest
   ```

---

## How It Works

- **Data Layer**: Ingests raw breach disclosures from the HHS OCR Breach Portal, cleans and normalizes them (entities, breach types, locations, states), and writes curated tables to `data/processed/` with derived temporal and severity features.
- **Intelligence Layer**: Implements analysis modules (trends, geography, entities, attack vectors) plus ML components for severity prediction, NLP attack classification, and Prophet-based forecasting, with metadata logged to `models/`.
- **Presentation Layer**: A multi-page Streamlit app (`app/`) surfaces trend, geographic, and risk insights with Plotly charts, Folium maps, and interactive filters tailored for security and compliance teams.

---

## Tech Stack

**Python • pandas • scikit-learn • XGBoost • SHAP • Prophet • Plotly • Folium • Streamlit • Pandera**

---

## Data Source

BreachIntel is built on the **U.S. Department of Health and Human Services (HHS) Office for Civil Rights (OCR) Breach Portal**, a public registry of breaches of unsecured protected health information reported under the HIPAA Breach Notification Rule.  
The dataset contains **7,400+ records** from **2009–present**, including covered entity name and type, state, breach type, location of breached information, and number of individuals affected.

- HHS OCR Breach Portal: <https://ocrportal.hhs.gov/ocr/breach/breach_report.jsf>

---

## Model Performance

> Model performance metrics will be published after training is complete.

---

## Project Structure

```text
breachintel/
├─ app/                     # Streamlit dashboard pages & components
├─ src/
│  └─ breachintel/
│     ├─ analysis/          # Trend, geographic, entity, attack-vector analyzers
│     ├─ data/              # Collector, cleaner, validator, feature engineer
│     ├─ ml/                # Severity model, NLP classifier, forecaster, risk scorer
│     ├─ utils/             # Config, logging, constants, caching
│     └─ visualization/     # Plotly charts and Folium map builders
├─ data/
│  ├─ raw/                  # Raw HHS OCR exports (gitignored)
│  ├─ processed/            # Cleaned tables (gitignored)
│  └─ sample/               # Small sample data checked into git
├─ models/                  # Trained artifacts & metadata (gitignored)
├─ tests/                   # Pytest suite for data, ML, and analytics
├─ notebooks/               # Exploratory analysis and prototyping
├─ scripts/                 # CLI utilities (download, train, etc.)
└─ docs/                    # Architecture, model cards, screenshots
```

---

## Documentation

- [ARCHITECTURE.md](docs/ARCHITECTURE.md)
- [MODEL_CARD.md](docs/MODEL_CARD.md)
- [CONTRIBUTING.md](CONTRIBUTING.md)

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Author

**Yewku Enchill-Yawson** — Biomedical Scientist · Healthcare Cybersecurity & Data Intelligence  
Research Assistant, Noguchi Memorial Institute for Medical Research (NMIMR)  
BSc Biomedical Science, University of Cape Coast

---

## Acknowledgments

- **HHS Office for Civil Rights** for maintaining and publishing the healthcare breach dataset that underpins this work.

