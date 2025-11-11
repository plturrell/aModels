FROM amodels/training-base:latest

WORKDIR /app

# Copy the server binary
COPY cmd/server/server /app/server
COPY cmd/server/ui /app/ui

# Expose port
EXPOSE 4173

# Run the server
CMD ["/app/server", "-addr", "0.0.0.0:4173"]

