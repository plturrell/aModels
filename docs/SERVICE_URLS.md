# Service URLs and Access Information

## ✅ External Access (Public IP: 54.196.0.75)

All services are now accessible via external IP after AWS Security Group configuration.

### Core Services

| Service | HTTP URL | Port | Status | Notes |
|---------|----------|------|--------|-------|
| **Neo4j Browser** | http://54.196.0.75:7474 | 7474 | ✅ Active | Username: `neo4j`, Password: `amodels123` |
| **Neo4j Bolt** | bolt://ec2-54-196-0-75.compute-1.amazonaws.com:7687 | 7687 | ✅ Active | Database connection |
| **Graph Service** | http://54.196.0.75:8080 | 8080 | ⚠️ Check | Graph computation API (may need to be started) |
| **Graph Admin** | http://54.196.0.75:19080 | 19080 | ⚠️ Check | Admin interface |
| **LocalAI** | http://54.196.0.75:8081 | 8081 | ✅ Accessible | OpenAI-compatible API (404 on root, but /v1/models works) |
| **Extract Service** | http://54.196.0.75:8082 | 8082 | ✅ Accessible | Data extraction API (404 on root, but /healthz works) |
| **Extract gRPC** | - | 9090 | ⚠️ Check | gRPC endpoint |
| **Extract Arrow Flight** | - | 8815 | ⚠️ Check | Arrow Flight endpoint |
| **Search Inference** | http://54.196.0.75:8090 | 8090 | ✅ Accessible | Search API (404 on root is normal) |
| **Search Python** | http://54.196.0.75:8091 | 8091 | ⚠️ Check | Python search service |
| **AgentFlow** | http://54.196.0.75:8001 | 8001 | ✅ Accessible | Workflow orchestration (404 on root, /healthz works) |
| **Browser Automation** | http://54.196.0.75:8070 | 8070 | ✅ Accessible | Browser automation API (404 on root, /healthz works) |
| **Elasticsearch** | http://54.196.0.75:9200 | 9200 | ⚠️ Check | Search engine |
| **Postgres** | - | 5432 | ⚠️ Check | Database (requires client) |
| **Redis** | - | 6379 | ⚠️ Check | Cache (requires client) |

## Service Details

### Neo4j Browser
- **URL**: http://54.196.0.75:7474
- **Bolt URL**: bolt://ec2-54-196-0-75.compute-1.amazonaws.com:7687
- **Credentials**: 
  - Username: `neo4j`
  - Password: `amodels123`
- **Status**: ✅ Confirmed working

### Graph Service
- **Base URL**: http://54.196.0.75:8080
- **Graph Endpoint**: http://54.196.0.75:8080/graph
- **Admin Endpoint**: http://54.196.0.75:19080
- **Purpose**: Graph computation and queries

### LocalAI
- **Base URL**: http://54.196.0.75:8081
- **Models Endpoint**: http://54.196.0.75:8081/v1/models
- **Chat Endpoint**: http://54.196.0.75:8081/v1/chat/completions
- **Purpose**: OpenAI-compatible inference server

### Extract Service
- **Base URL**: http://54.196.0.75:8082
- **Health Check**: http://54.196.0.75:8082/healthz
- **Graph Endpoint**: http://54.196.0.75:8082/graph
- **Purpose**: Data extraction and schema replication

### Search Services
- **Search Inference**: http://54.196.0.75:8090
- **Search Python**: http://54.196.0.75:8091
- **Elasticsearch**: http://54.196.0.75:9200
- **Purpose**: Intelligent search and discovery

### AgentFlow
- **Base URL**: http://54.196.0.75:8001
- **Health Check**: http://54.196.0.75:8001/healthz
- **Flows Endpoint**: http://54.196.0.75:8001/flows
- **Purpose**: LangFlow workflow orchestration

### Browser Automation
- **Base URL**: http://54.196.0.75:8070
- **Health Check**: http://54.196.0.75:8070/healthz
- **Navigate Endpoint**: http://54.196.0.75:8070/navigate
- **Purpose**: Headless Chromium automation

## ✅ Confirmed Working Services

The following services have been tested and are accessible:

- ✅ **Neo4j Browser**: http://54.196.0.75:7474
- ✅ **LocalAI**: http://54.196.0.75:8081/v1/models
- ✅ **Extract Service**: http://54.196.0.75:8082/healthz
- ✅ **AgentFlow**: http://54.196.0.75:8001/healthz
- ✅ **Browser Automation**: http://54.196.0.75:8070/healthz
- ✅ **Search Inference**: http://54.196.0.75:8090

## Testing Connectivity

### From Browser
Open Neo4j Browser: http://54.196.0.75:7474

### From Command Line
```bash
# Test Neo4j
curl http://54.196.0.75:7474

# Test Graph Service
curl http://54.196.0.75:8080/healthz

# Test LocalAI
curl http://54.196.0.75:8081/v1/models

# Test Extract Service
curl http://54.196.0.75:8082/healthz

# Test AgentFlow
curl http://54.196.0.75:8001/healthz
```

### From Python
```python
import requests

# Neo4j
response = requests.get("http://54.196.0.75:7474")
print(response.json())

# Graph Service
response = requests.get("http://54.196.0.75:8080/healthz")
print(response.json())
```

## Security Notes

⚠️ **Important Security Considerations:**

1. **Change Default Passwords**: The Neo4j password `amodels123` should be changed in production
2. **IP Whitelist**: Consider restricting security group access to specific IPs
3. **Use HTTPS**: For production, set up SSL/TLS certificates
4. **VPN Access**: Consider requiring VPN before accessing services
5. **Rate Limiting**: Configure rate limiting to prevent abuse

## Troubleshooting

### Service Not Accessible

1. **Check if service is running:**
   ```bash
   docker ps | grep <service-name>
   ```

2. **Check service logs:**
   ```bash
   docker logs <service-name>
   ```

3. **Verify AWS Security Group:**
   - Go to AWS Console → EC2 → Security Groups
   - Verify inbound rules for the port
   - Check if your IP is allowed

4. **Test from inside the server:**
   ```bash
   docker exec <service-name> wget -q -O- http://localhost:<port>
   ```

### Connection Timeout

- Verify the service is running
- Check AWS Security Group rules
- Verify the port is correct
- Check if firewall is blocking the connection

## Alternative: SSH Tunnel

If you prefer not to expose services publicly, use SSH tunneling:

```bash
./scripts/setup_neo4j_tunnel.sh
# or
./scripts/setup_all_tunnels.sh
```

Then access services at `http://localhost:PORT`.

