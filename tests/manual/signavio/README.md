# Manual Signavio Integration Fixtures

This directory contains stub artifacts for testing the bidirectional Signavio integration without
calling the real APIs.

## Files

- `sample_ingestion.csv` – simple case-centric event log that mimics the payload sent through the
  Signavio Ingestion API (CSV flavour).
- `sample_ingestion.avsc` – Apache Avro schema describing the CSV structure and matching the
  supported logical types from the Signavio documentation.

## Manual Testing With the Stub Client

1. **Run a stub upload**

   ```go
   ctx := context.Background()
   client := signavio.NewStubClient("manual")

   id, err := client.Upload(ctx, signavio.UploadRequest{
       Dataset:    "orders",
       FilePath:   "testing/manual/signavio/sample_ingestion.csv",
       SchemaPath: "testing/manual/signavio/sample_ingestion.avsc",
       PrimaryKeys: []string{"case_id", "event_time"},
   })
   ```

   The stub returns a deterministic identifier such as:

   ```text
   stub://signavio/uploads/orders/sample_ingestion.csv
   ```

2. **Fetch a stub OData view**

   ```go
   result, err := client.FetchOData(ctx, signavio.ODataQuery{
       ViewName:     "CaseOverview",
       SelectFields: []string{"case_id", "cycle_time"},
       Filter:       "cycle_time gt 0",
   })
   ```

   The stub yields a synthetic handle (for example
   `stub://signavio/odata/CaseOverview?fields=2`).

3. **Manual validation**

   - Inspect `sample_ingestion.csv` to confirm delimiter usage and canonical field names.
   - Update `sample_ingestion.avsc` when adding columns, keeping logical types
     (`timestamp-millis`, unions for nullable fields) in sync with Signavio expectations.
   - Extend the stub client or create additional fixtures here as integration needs evolve.

These assets allow developers to iterate on orchestration logic and wiring without Signavio access.

## DeepAgents / Agent Workflow Usage

The stub tools are exposed to DeepAgents via the `signavio_stub_upload` and
`signavio_stub_fetch_view` tools. Example conversation inputs:

- Upload telemetry captured in `agent_telemetry.csv`:

  ```text
  Use the signavio_stub_upload tool with dataset="agent-logs",
  file_path="agent_telemetry.csv", schema_path="agent_telemetry.avsc",
  primary_keys=["agent_run_id", "task_id"].
  ```

- Retrieve the current process library and feed it to downstream analysis:

  ```text
  Call signavio_stub_fetch_view with view_name="ProcessLibrary" then summarize
  the returned process metrics.
  ```

These instructions keep agent workflows aligned with the manual-first testing
strategy until live Signavio endpoints are available.
