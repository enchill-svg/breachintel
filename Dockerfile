FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential curl && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN python -m pip install --upgrade pip && \
    python -m pip install -r requirements.txt

COPY src/ ./src/
COPY app/ ./app/
COPY data/sample/ ./data/sample/
COPY models/ ./models/
COPY .streamlit/ ./.streamlit/
COPY pyproject.toml .

RUN python -m pip install -e .

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --retries=3 CMD curl -f http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "app/Home.py", "--server.port=8501", "--server.address=0.0.0.0"]

