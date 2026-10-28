FROM python:3.12-slim

# libgomp: OpenMP runtime needed by xgboost / lightgbm
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY config.py app.py ./
COPY src/ src/
COPY dashboard/ dashboard/
COPY artifacts/best_model.joblib artifacts/underwriting_benchmarks.json artifacts/
COPY data/demo_batch_300.csv data/example_applicants.csv data/template_applicants.csv data/
COPY data/test_files/ data/test_files/

RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

ENV HOST=0.0.0.0 PORT=8000
EXPOSE 8000
HEALTHCHECK CMD python -c "import urllib.request,os; urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/api/health')"
CMD ["sh", "-c", "uvicorn app:app --host $HOST --port $PORT"]
