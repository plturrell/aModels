# syntax=docker/dockerfile:1.6
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy service code
COPY services/agentflow/service/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY services/agentflow/service ./service
COPY services/agentflow/flows ./flows
COPY services/agentflow/store ./store

# Set environment
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

EXPOSE 8001

# Run FastAPI service
CMD ["uvicorn", "service.main:app", "--host", "0.0.0.0", "--port", "8001"]

