# syntax=docker/dockerfile:1.6
FROM mcr.microsoft.com/playwright/python:v1.48.0-noble

WORKDIR /app

# Install Python dependencies
COPY services/browser/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Install Playwright browsers
RUN playwright install chromium
RUN playwright install-deps chromium

# Copy browser automation service
COPY services/browser/automation.py ./

# Set environment
ENV PYTHONUNBUFFERED=1 \
    PLAYWRIGHT_BROWSERS_PATH=/ms-playwright

EXPOSE 8070

# Run FastAPI service
CMD ["uvicorn", "automation:app", "--host", "0.0.0.0", "--port", "8070"]

