VENV_DIR := .venv
PYTHON := $(VENV_DIR)/Scripts/python
PIP := $(VENV_DIR)/Scripts/pip

.PHONY: setup data train test lint format run sample docker-build docker-run clean all

setup:
	python -m venv $(VENV_DIR)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	$(PIP) install -r requirements-dev.txt
	$(PIP) install -e .
	if not exist .env copy .env.example .env
	$(VENV_DIR)/Scripts/pre-commit install

data:
	$(PYTHON) scripts/download_data.py
	$(PYTHON) -m breachintel.data.cleaner
	$(PYTHON) -m breachintel.data.validator
	$(PYTHON) -m breachintel.data.feature_engineer

train:
	$(PYTHON) scripts/train_models.py

test:
	$(PYTHON) -m pytest

lint:
	$(PYTHON) -m ruff check src/ tests/ app/
	$(PYTHON) -m ruff format --check src/ tests/ app/

format:
	$(PYTHON) -m ruff check --fix src/ tests/ app/
	$(PYTHON) -m ruff format src/ tests/ app/

run:
	$(PYTHON) -m streamlit run app/Home.py

sample:
	$(PYTHON) scripts/generate_sample.py

docker-build:
	docker build -t breachintel:latest .

docker-run:
	docker run -p 8501:8501 breachintel:latest

clean:
	del /Q data\\raw\\*.csv 2> NUL || exit /B 0
	del /Q data\\processed\\*.csv 2> NUL || exit /B 0
	del /Q models\\*.joblib 2> NUL || exit /B 0
	del /Q models\\*.json 2> NUL || exit /B 0
	rmdir /S /Q __pycache__ 2> NUL || exit /B 0
	rmdir /S /Q .pytest_cache 2> NUL || exit /B 0
	rmdir /S /Q dist 2> NUL || exit /B 0
	rmdir /S /Q build 2> NUL || exit /B 0

all: data train test run
