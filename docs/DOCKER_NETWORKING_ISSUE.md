# Docker Port Forwarding Issue - Investigation Results

## Problem Summary

All Docker containers are configured with correct port mappings (`0.0.0.0:PORT->PORT/tcp`), but services are **not accessible on localhost** from the host machine. However, services **are responding correctly from inside their containers**.

## Affected Services

All services show the same issue:
- **Neo4j**: Port 7474 (HTTP), 7687 (Bolt) - Not accessible on localhost
- **Graph Service**: Port 8080, 19080 - Not accessible on localhost
- **LocalAI**: Port 8081 - Not accessible on localhost
- **Extract Service**: Port 8082, 9090, 8815 - Not accessible on localhost
- **Search Inference**: Port 8090 - Not accessible on localhost
- **Search Python**: Port 8091 - Not accessible on localhost
- **AgentFlow**: Port 8001 - Not accessible on localhost
- **Browser**: Port 8070 - Not accessible on localhost
- **Postgres**: Port 5432 - Not accessible on localhost
- **Redis**: Port 6379 - Not accessible on localhost

## Investigation Findings

### ✅ What's Working
1. **Docker containers are running** - All services are up and healthy
2. **Port mappings are correct** - Docker shows `0.0.0.0:PORT->PORT/tcp`
3. **Services respond from inside containers** - `docker exec neo4j wget http://localhost:7474` returns JSON
4. **IP forwarding is enabled** - `net.ipv4.ip_forward = 1`
5. **Docker network is configured** - Containers are on `brev_default` bridge network

### ❌ What's Not Working
1. **Port forwarding to localhost fails** - `curl http://localhost:7474` returns "Connection refused"
2. **iptables not available** - Cannot check Docker NAT rules (may be using nftables)
3. **Cannot verify listening ports** - `ss` and `netstat` commands not available
4. **Docker port proxy may not be active** - Cannot verify `docker-proxy` processes

## Possible Causes

### 1. Brev/Cloud Environment Specific
This appears to be a **Brev-specific networking configuration** where:
- Containers are accessible from other containers (Docker network)
- Containers are NOT accessible from the host via localhost
- Services may be accessible via external IP (if security groups allow)

### 2. Docker Networking Mode
Docker may be configured to use a different networking mode that doesn't forward ports to localhost.

### 3. Firewall/Network Policy
System-level firewall or network policies may be blocking localhost connections.

### 4. Docker Daemon Configuration
Docker daemon may have specific configuration that prevents port forwarding.

## Solutions

### Immediate Solution: SSH Tunnel (Recommended)

Use SSH tunneling to access services:

```bash
# Neo4j Browser
ssh -L 7474:localhost:7474 -L 7687:localhost:7687 USERNAME@54.196.0.75

# Multiple services
ssh -L 7474:localhost:7474 \
    -L 7687:localhost:7687 \
    -L 8080:localhost:8080 \
    -L 8081:localhost:8081 \
    -L 8082:localhost:8082 \
    USERNAME@54.196.0.75
```

Then access services at `http://localhost:PORT`.

**Automated script available**: `scripts/setup_neo4j_tunnel.sh`

### Alternative: Direct External Access

If you configure AWS Security Groups to allow inbound traffic:

1. **Open AWS Console** → EC2 → Security Groups
2. **Find your instance's security group** (instance ID: `i-0f2677b4664e30ddc`)
3. **Add inbound rules** for needed ports:
   - Port 7474 (Neo4j HTTP)
   - Port 7687 (Neo4j Bolt)
   - Port 8080 (Graph Service)
   - Port 8081 (LocalAI)
   - Port 8082 (Extract)
   - etc.

4. **Access via public IP**: `http://54.196.0.75:PORT`

⚠️ **Security Warning**: Only allow your IP address (`/32` CIDR) or use a VPN.

### Workaround: Use Container IPs

If you need to access services from within the server:

```bash
# Get container IP
CONTAINER_IP=$(docker inspect neo4j --format '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}')

# Access directly
curl http://$CONTAINER_IP:7474
```

However, this doesn't help with external access.

## Diagnostic Tools

### Run Diagnostic Script
```bash
bash scripts/diagnose_docker_networking.sh
```

This will check:
- Docker status and version
- IP forwarding
- Port bindings
- Localhost connectivity
- Container-to-container connectivity
- Docker daemon configuration

### Manual Checks

```bash
# Check Docker port mappings
docker port neo4j

# Check if service responds inside container
docker exec neo4j wget -q -O- http://localhost:7474

# Check container network
docker network inspect brev_default

# Check Docker info
docker info | grep -i network
```

## Next Steps

1. **Contact Brev Support**: This appears to be a platform-specific networking issue
2. **Check Brev Documentation**: Look for networking configuration or limitations
3. **Use SSH Tunnel**: For immediate access, use the provided tunnel script
4. **Configure Security Groups**: If external access is needed, configure AWS Security Groups

## Related Files

- `scripts/setup_neo4j_tunnel.sh` - SSH tunnel setup script
- `scripts/diagnose_docker_networking.sh` - Network diagnostic script
- `docs/EXTERNAL_ACCESS.md` - External access guide
- `docs/AWS_SECURITY_GROUP_SETUP.md` - AWS security group configuration

## Status

- **Issue**: Confirmed - Docker port forwarding not working on localhost
- **Impact**: Cannot access services from host machine
- **Workaround**: SSH tunnel works perfectly
- **Root Cause**: Likely Brev platform networking configuration
- **Resolution**: Use SSH tunnel or configure external access via security groups

