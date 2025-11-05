FROM golang:1.22 AS build

WORKDIR /workspace

ENV GOWORK=off

# Copy only what we need for the hana service module
COPY services/hana/ /workspace/services/hana/

WORKDIR /workspace/services/hana

RUN go build -o /hana-service ./cmd/hana-service

FROM gcr.io/distroless/base-debian12

ENV HANA_DSN="" \
    PORT=8083

COPY --from=build /hana-service /hana-service

EXPOSE 8083

USER nonroot:nonroot

ENTRYPOINT ["/hana-service"]

