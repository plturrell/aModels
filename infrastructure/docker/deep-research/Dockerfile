FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential curl && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade uv

COPY models/open_deep_research/uv.lock models/open_deep_research/pyproject.toml ./

RUN uv pip sync --system --frozen uv.lock

RUN pip install "langgraph-cli[inmem]>=0.3.1"

COPY models/open_deep_research/ .

EXPOSE 2024

CMD ["langgraph", "dev", "--config", "langgraph.json", "--host", "0.0.0.0", "--port", "2024", "--allow-blocking"]
