# syntax=docker/dockerfile:1.6
FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    GO_VERSION=1.22.6

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv \
    git build-essential curl pkg-config \
    libssl-dev libffi-dev && \
    rm -rf /var/lib/apt/lists/*

RUN curl -fsSL https://go.dev/dl/go${GO_VERSION}.linux-amd64.tar.gz | tar -C /usr/local -xz && \
    ln -s /usr/local/go/bin/go /usr/local/bin/go && \
    ln -s /usr/bin/python3 /usr/local/bin/python

RUN pip install --no-cache-dir pip==24.2

CMD ["bash"]
