# Build stage
FROM golang:1.22-alpine AS builder

WORKDIR /app/agenticAiETH_layer4_Extract

# Copy go.mod
COPY agenticAiETH_layer4_Extract/go.mod .

# Download dependencies (this will be a no-op for now)
RUN go mod download

# Copy the rest of the source code
COPY agenticAiETH_layer4_Extract/ .

# Build the application
RUN go build -o /server ./main.go

# Final stage
FROM alpine:latest

# Install Python runtime for helper scripts
RUN apk add --no-cache python3 py3-pip

WORKDIR /app

# Copy the built executable and helper scripts from the builder stage
COPY --from=builder /server ./server
COPY --from=builder /app/agenticAiETH_layer4_Extract/scripts ./scripts

# Install Python dependencies required by the helper scripts
RUN pip3 install --no-cache-dir -r ./scripts/requirements.txt

# Command to run the service with dependency waits
ENTRYPOINT ["./scripts/start_extract.sh"]
