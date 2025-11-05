# Research Reports Data Lifecycle

Last updated: 2025-11-05

Open Deep Research now persists responses in the `research_reports` table (see migration `004_create_research_reports.sql`). To operate this feature safely, adopt the following retention and anonymisation policies.

## Retention Policy

- **Raw JSON (`report_json`)**: retains full model output, which may include sensitive details provided by operators. Keep for **90 days** in non-production and **30 days** in production unless legal/regulatory requirements dictate otherwise.
- **Summaries (`report_summary`)**: can be kept longer for catalog UX (up to 1 year). Consider trimming to 500 characters to minimise exposure.
- **Metadata (`topic`, `data_element_id`)**: acts as an index and should align with catalog lifecycle rules. Delete when the parent data element is removed.

### Recommended Implementation

1. **Go clean-up task** – add a scheduled job in `services/catalog/research/store.go` that executes daily (cron or background goroutine) and runs:
   ```sql
   DELETE FROM research_reports
     WHERE created_at < NOW() - INTERVAL '30 days';
   ```
   Wrap in configuration so retention windows are adjustable via env vars.
2. **Manual command** – expose a CLI (e.g., `go run ./cmd/migrate retention-cleanup`) for operators to trigger retention trimming on demand.
3. **Monitoring** – add a Prometheus metric or log entry whenever rows are purged to confirm the job is active.

## Anonymisation Guidelines

- **PII Scrubbing**: Before inserting into `research_reports`, run the summary/sections through a redaction helper that masks email addresses, phone numbers, and known PII patterns. Add this as a helper in `services/catalog/research/store.go`.
- **Context Control**: Avoid storing raw tool outputs that include authentication tokens or sensitive connection details. Ensure Deep Research agent prompts instruct models not to echo secrets.
- **Subject Access Requests**: Document a process to export or delete specific `report_json` rows when requested. Provide an SQL snippet in the runbook:
  ```sql
  SELECT id, created_at
    FROM research_reports
   WHERE topic ILIKE '%<identifier>%';
  DELETE FROM research_reports WHERE id = <id>;
  ```

## Backup & Restore

- Backups created with `pg_dump` will include `research_reports`. Confirm retention jobs run before backups to minimise dataset size.
- During restore, run Goose migrations first, then import data. If backups pre-date the table, Goose will create it automatically.

## Operator Checklist

- [ ] Set retention env vars (e.g., `RESEARCH_REPORT_RETENTION_DAYS=30`; default is 30 days, set to `0` to disable automatic pruning).
- [ ] Enable the clean-up job in production deployments.
- [ ] Add monitoring/alerting when the table grows unexpectedly.
- [ ] Ensure documentation for subject access requests is linked in the team runbook.
