## HANA service (go-hdb)

Minimal Go microservice exposing `/sql` over HTTP using `github.com/SAP/go-hdb/driver`.

### Configuration

Set environment variables:

- `HANA_DSN` – DSN string, e.g.: `hdb://user:pass@host:39015`
- `HANA_MAX_OPEN_CONNS` – optional, default `5`
- `PORT` – optional, default `8083`

### Endpoints

- `POST /sql` – body: `{ "query": string, "args": any[] }`
  - Returns: `{ "rows": [ { col: value, ... } ] }`

### Build & Run

```bash
cd hana
go build -o bin/hana-service ./cmd/hana-service
HANA_DSN="hdb://user:pass@host:39015" ./bin/hana-service
```


