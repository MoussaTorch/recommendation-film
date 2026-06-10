FROM python:3.10-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir numpy==1.26.4 cython \
    && pip install --no-cache-dir scikit-surprise==1.1.3 --no-build-isolation \
    && pip install --no-cache-dir "setuptools<70" \
    && pip install --no-cache-dir -r requirements.txt

COPY settings/ settings/
COPY src/ src/
COPY interface/api/ interface/api/

RUN mkdir -p data/interim data/processed models metrics

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health')"

CMD ["uvicorn", "interface.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
