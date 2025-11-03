# Build stage
FROM golang:1.23-alpine AS builder

WORKDIR /app/services/extract

# Copy go.mod
COPY services/extract/go.mod .

# Download dependencies
RUN go mod download

# Copy the rest of the source code
COPY services/extract/ .

# Build the application
RUN go build -o /server ./main.go

# Final stage
FROM alpine:latest

# Install Python runtime for helper scripts
RUN apk add --no-cache python3 py3-pip

WORKDIR /app

# Copy the built executable and helper scripts from the builder stage
COPY --from=builder /server ./server
COPY --from=builder /app/services/extract/scripts ./scripts

# Install Python dependencies required by the helper scripts
RUN pip3 install --no-cache-dir -r ./scripts/requirements.txt

# Command to run the service with dependency waits
ENTRYPOINT ["./scripts/start_extract.sh"]
