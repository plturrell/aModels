FROM golang:1.22 as build
WORKDIR /app
COPY . ./
RUN cd cmd/hana-service && go build -o /hana-service

FROM gcr.io/distroless/base-debian12
ENV HANA_DSN="" PORT=8083
COPY --from=build /hana-service /hana-service
EXPOSE 8083
USER nonroot:nonroot
ENTRYPOINT ["/hana-service"]


