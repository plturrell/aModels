FROM python:3.10-slim

WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY services/training/requirements-service.txt /workspace/requirements-service.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements-service.txt

# Copy training service code
COPY services/training/ /workspace/services/training/

# Set PYTHONPATH to include services directory
ENV PYTHONPATH=/workspace

# Expose port
EXPOSE 8080

# Set working directory to training service
WORKDIR /workspace/services/training

# Run the service
CMD ["python3", "main.py"]

