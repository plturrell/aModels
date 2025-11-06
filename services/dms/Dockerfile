FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100

WORKDIR /app

COPY services/dms/requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY services/dms/app ./app

ENV DMS_APP_NAME="aModels DMS" \
    DMS_DEBUG=false

EXPOSE 8080

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
