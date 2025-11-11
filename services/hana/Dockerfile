FROM golang:1.22 AS build

WORKDIR /workspace

ENV GOWORK=off

# Copy only what we need for the hana service module
COPY services/hana/ /workspace/services/hana/

WORKDIR /workspace/services/hana

RUN go build -o /hana-service ./cmd/hana-service

FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends ca-certificates && \
    rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --shell /usr/sbin/nologin appuser

WORKDIR /home/appuser

COPY --from=build /hana-service /usr/local/bin/hana-service

ENV HANA_DSN="" \
    PORT=8083

EXPOSE 8083

USER appuser

ENTRYPOINT ["/usr/local/bin/hana-service"]
