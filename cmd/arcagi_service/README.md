# ARC-AGI Query Service

This service provides a RESTful API for querying various AI benchmark datasets, including ARC-AGI, BoolQ, and HellaSwag, stored in a SAP HANA database.

## Purpose

The primary purpose of this service is to provide a standardized, efficient way for training scripts and other tools to access benchmark data. By centralizing data access, we can ensure consistency and simplify the process of running experiments.

## Configuration

The service is configured via environment variables. It can also load variables from a `.env` file located in the project root.

**Required Environment Variables:**

*   `HANA_HOST`: The hostname of the HANA database.
*   `HANA_PORT`: The port number for the HANA database.
*   `HANA_USER`: The username for the HANA database.
*   `HANA_PASSWORD`: The password for the HANA database.
*   `ARCAGI_SERVICE_PORT`: The port on which the service will run (defaults to `8090`).

## Running the Service

```bash
go run ./cmd/arcagi_service/main.go
```

## API Endpoints

The service exposes a versioned API under the `/api/v1` prefix.

### Health Check

*   `GET /health`: Checks the health of the service and its database connection.

### ARC-AGI & ARC-AGI-2

*   `GET /api/v1/arcagi/tasks`: Lists all ARC tasks with pagination.
*   `GET /api/v1/arcagi/tasks/{taskId}`: Retrieves a specific ARC task.
*   `GET /api/v1/arcagi/tasks/{taskId}/samples`: Retrieves the training/test samples for a task.
*   `GET /api/v1/arcagi/tasks/{taskId}/grids`: Retrieves the raw grid data for a task.
*   `GET /api/v1/arcagi/stats`: Retrieves statistics about the ARC dataset.

(The same endpoints are available under `/api/v1/arcagi2` for the ARC-AGI-2 dataset.)

### BoolQ

*   `GET /api/v1/boolq/questions`: Lists questions from the BoolQ dataset.
*   `GET /api/v1/boolq/stats`: Retrieves statistics about the BoolQ dataset.

### HellaSwag

*   `GET /api/v1/hellaswag/examples`: Lists examples from the HellaSwag dataset.
*   `GET /api/v1/hellaswag/stats`: Retrieves statistics about the HellaSwag dataset.

**Query Parameters:**

Most listing endpoints support `limit` and `offset` for pagination.
