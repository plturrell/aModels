FROM python:3.13-slim-bullseye

ENV DEBIAN_FRONTEND=noninteractive
ENV EXIFTOOL_PATH=/usr/bin/exiftool
ENV FFMPEG_PATH=/usr/bin/ffmpeg

# Runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    exiftool \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy markitdown package
COPY third_party/markitdown /app/markitdown

# Install markitdown
RUN pip --no-cache-dir install /app/markitdown/packages/markitdown[all] \
    /app/markitdown/packages/markitdown-sample-plugin

# Create HTTP service wrapper
COPY services/markitdown-service /app/service
WORKDIR /app/service

# Install service dependencies
RUN pip --no-cache-dir install flask>=3.0.0 werkzeug>=3.0.0

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/healthz')"

# Run service
CMD ["python", "server.py"]

