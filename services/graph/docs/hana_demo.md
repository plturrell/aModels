# Graph + HANA Demo

The Go runtime works end-to-end with the HANA repository. Once a HANA instance is available:

```bash
# ensure the hana build tag is active and environment variables are set
export CGO_ENABLED=1
export HANA_HOST=...
export HANA_USER=...
export HANA_PASSWORD=...
export HANA_SCHEMA=A2A

# compile with hana support and run the integrated demo
go run -tags hana ./cmd/langgraph demo -checkpoint hana -mode sync -input 2
```

The demo performs the following steps:

1. Connects to HANA using the repository layer (`pkg/repository`).
2. Uses `pkg/unified_db` to initialise schemas and persist trail data.
3. Executes the Go graph runtime with checkpointing enabled, writing checkpoints to HANA.
4. Prints node outputs and the final default result.

The same invocation is exercised in CI via `make test-hana` (see `Makefile`).
