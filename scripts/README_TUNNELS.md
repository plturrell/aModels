# SSH Tunnel Scripts for Service Access

## Problem

Docker port forwarding is not working on localhost in this Brev environment. All services are running correctly inside containers, but cannot be accessed from the host machine via `localhost:PORT`.

## Solution: SSH Tunnels

Use SSH tunnels to access services from your local machine.

## Quick Start

### Neo4j Only

```bash
./scripts/setup_neo4j_tunnel.sh
```

Then open: `http://localhost:7474`

### All Services

```bash
./scripts/setup_all_tunnels.sh
```

This sets up tunnels for:
- Neo4j (7474, 7687)
- Graph Service (8080, 19080)
- LocalAI (8081)
- Extract (8082, 9090, 8815)
- Search (8090, 8091)
- AgentFlow (8001)
- Browser (8070)
- Postgres (5432)
- Redis (6379)
- Elasticsearch (9200)

## Configuration

Set environment variables before running:

```bash
export SERVER_IP=54.196.0.75
export SSH_USER=your_username
./scripts/setup_neo4j_tunnel.sh
```

Or for Neo4j-specific script:

```bash
export NEO4J_SERVER_IP=54.196.0.75
export NEO4J_SSH_USER=your_username
./scripts/setup_neo4j_tunnel.sh
```

## Manual SSH Tunnel

If you prefer to set up tunnels manually:

```bash
ssh -L 7474:localhost:7474 \
    -L 7687:localhost:7687 \
    -L 8080:localhost:8080 \
    -L 8081:localhost:8081 \
    USERNAME@54.196.0.75
```

## Stopping Tunnels

Press `Ctrl+C` in the terminal running the tunnel, or:

```bash
pkill -f 'ssh.*7474:localhost:7474'
```

## Troubleshooting

### Port Already in Use

If a port is already in use, the script will ask if you want to kill the process. You can also manually free the port:

```bash
# Find process using port
lsof -i :7474

# Kill process
kill -9 <PID>
```

### Connection Refused

- Verify SSH access to the server works
- Check that services are running: `docker ps`
- Verify ports are correct in docker-compose.yml

### Diagnostic

Run the diagnostic script to check Docker networking:

```bash
./scripts/diagnose_docker_networking.sh
```

## Accessing Services

Once tunnels are active, access services at:

- **Neo4j Browser**: http://localhost:7474
  - Username: `neo4j`
  - Password: `amodels123`

- **Graph Service**: http://localhost:8080
- **LocalAI**: http://localhost:8081
- **Extract Service**: http://localhost:8082
- **Search Inference**: http://localhost:8090
- **AgentFlow**: http://localhost:8001
- **Browser Automation**: http://localhost:8070
- **Elasticsearch**: http://localhost:9200

## Related Documentation

- `docs/DOCKER_NETWORKING_ISSUE.md` - Detailed investigation of the issue
- `docs/EXTERNAL_ACCESS.md` - External access options
- `docs/AWS_SECURITY_GROUP_SETUP.md` - AWS security group configuration

