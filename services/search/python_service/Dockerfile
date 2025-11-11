# syntax=docker/dockerfile:1.6
FROM python:3.10-slim

WORKDIR /workspace

# Install dependencies first to leverage Docker layer caching
COPY services/search/python_service/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy application sources
COPY services/search/python_service/ ./

EXPOSE 8091

CMD ["python", "app.py"]
