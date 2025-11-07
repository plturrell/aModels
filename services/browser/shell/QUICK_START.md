# Quick Start: Backend Connection

## Fastest Way to Get Backend Running

### Option 1: Use the Startup Script (Recommended)

```bash
cd services/browser/shell
./start-backend.sh
```

This will:
- Start Gateway service on port 8000
- Start Shell Server on port 4173
- Wait for services to be ready
- Show status and logs

### Option 2: Manual Start

#### Terminal 1: Gateway Service
```bash
cd services/gateway
export GATEWAY_PORT=8000
./start.sh
# Or: uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

#### Terminal 2: Shell Server
```bash
cd services/browser/shell
export SHELL_GATEWAY_URL=http://localhost:8000
export VITE_SHELL_API=""
./cmd/server/server -addr :4173
```

#### Terminal 3: Frontend (Development)
```bash
cd services/browser/shell/ui
export VITE_SHELL_API=""
npm run dev
```

## Verify Connection

### 1. Test Gateway
```bash
curl http://localhost:8000/healthz
```

### 2. Test Shell Server
```bash
curl http://localhost:4173/api/localai/models
```

### 3. Test Search Endpoint
```bash
curl http://localhost:4173/search/unified \
  -X POST \
  -H "Content-Type: application/json" \
  -d '{"query": "test", "top_k": 5}'
```

## Frontend Configuration

The frontend uses `VITE_SHELL_API` environment variable:
- **Empty string** (`""`) = Relative paths (uses shell server)
- **URL** = Direct connection to that URL

For development with shell server:
```bash
export VITE_SHELL_API=""
```

For direct gateway connection (bypass shell server):
```bash
export VITE_SHELL_API="http://localhost:8000"
```

## Common Issues

### Port Already in Use
```bash
# Find process using port 8000
lsof -i :8000
# Kill it
kill -9 <PID>
```

### Gateway Not Starting
```bash
# Check if dependencies are installed
cd services/gateway
pip install -r requirements.txt
```

### Shell Server Not Building
```bash
cd services/browser/shell/cmd/server
go build -o server main.go
```

## Next Steps

1. ✅ Backend services running
2. ✅ Test search functionality
3. ✅ Test narrative generation
4. ✅ Test dashboard generation
5. ✅ Test PowerPoint export

