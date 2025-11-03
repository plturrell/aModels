# Build stage
FROM golang:1.25.3-alpine AS builder

WORKDIR /app

# Copy the required projects
COPY agenticAiETH_layer1_Blockchain ./agenticAiETH_layer1_Blockchain
COPY agenticAiETH_layer4_HANA ./agenticAiETH_layer4_HANA
COPY agenticAiETH_layer4_Graph ./agenticAiETH_layer4_Graph
COPY third_party/go-redis ./third_party/go-redis
COPY third_party/go-hdb ./third_party/go-hdb

# Set the working directory to the main project
WORKDIR /app/agenticAiETH_layer4_Graph

# Download dependencies
RUN go mod download

# Build the application
RUN go build -o /server ./cmd/graph-server

# Final stage
FROM alpine:latest

# Copy the built executable from the builder stage
COPY --from=builder /server /server

# Expose any port the server might be listening on (I'll assume 8080 for now)
EXPOSE 8080

# Command to run the executable
CMD ["/server"]
