#!/usr/bin/env bash
# Docker Networking Diagnostic Script
# Diagnoses Docker port forwarding issues

set -euo pipefail

echo "=== Docker Networking Diagnostic ==="
echo ""

# Check Docker status
echo "1. Docker Status:"
systemctl is-active docker >/dev/null 2>&1 && echo "   ✓ Docker is running" || echo "   ✗ Docker is not running"
echo ""

# Check Docker version
echo "2. Docker Version:"
docker version --format '{{.Server.Version}}' 2>/dev/null || echo "   Cannot get version"
echo ""

# Check IP forwarding
echo "3. IP Forwarding:"
if [ "$(sysctl -n net.ipv4.ip_forward)" = "1" ]; then
    echo "   ✓ IP forwarding is enabled"
else
    echo "   ✗ IP forwarding is disabled"
fi
echo ""

# Check Docker network
echo "4. Docker Networks:"
docker network ls
echo ""

# Check port bindings
echo "5. Container Port Bindings:"
docker ps --format "{{.Names}}" | while read container; do
    echo "   $container:"
    docker port "$container" 2>/dev/null | sed 's/^/     /' || echo "     No ports mapped"
done
echo ""

# Test localhost connectivity
echo "6. Testing Localhost Connectivity:"
for port in 7474 8080 8081 8082 8090 8091; do
    if timeout 1 bash -c "echo > /dev/tcp/localhost/$port" 2>/dev/null; then
        echo "   ✓ Port $port is accessible"
    else
        echo "   ✗ Port $port is NOT accessible"
    fi
done
echo ""

# Check iptables (if available)
echo "7. Docker iptables Rules:"
if command -v iptables >/dev/null 2>&1; then
    echo "   NAT rules (first 10):"
    sudo iptables -t nat -L DOCKER -n 2>/dev/null | head -10 | sed 's/^/     /' || echo "     Cannot read iptables"
else
    echo "   iptables not available"
fi
echo ""

# Check listening ports
echo "8. Listening Ports:"
if command -v ss >/dev/null 2>&1; then
    ss -tlnp 2>/dev/null | grep -E "7474|8080|8081|8082|8090|8091" | sed 's/^/     /' || echo "     No matching ports found"
elif command -v netstat >/dev/null 2>&1; then
    netstat -tlnp 2>/dev/null | grep -E "7474|8080|8081|8082|8090|8091" | sed 's/^/     /' || echo "     No matching ports found"
else
    echo "     Cannot check (ss/netstat not available)"
fi
echo ""

# Test container connectivity
echo "9. Container-to-Container Connectivity Test:"
if docker ps --format "{{.Names}}" | grep -q neo4j; then
    CONTAINER_IP=$(docker inspect neo4j --format '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' 2>/dev/null)
    if [ -n "$CONTAINER_IP" ]; then
        echo "   Neo4j container IP: $CONTAINER_IP"
        if docker exec neo4j wget -q -O- http://localhost:7474 >/dev/null 2>&1; then
            echo "   ✓ Neo4j is responding from inside container"
        else
            echo "   ✗ Neo4j is NOT responding from inside container"
        fi
    fi
fi
echo ""

# Check Docker daemon config
echo "10. Docker Daemon Configuration:"
if [ -f /etc/docker/daemon.json ]; then
    echo "   /etc/docker/daemon.json exists:"
    cat /etc/docker/daemon.json 2>/dev/null | sed 's/^/     /' || echo "     Cannot read"
else
    echo "   /etc/docker/daemon.json does not exist"
fi
echo ""

echo "=== Diagnostic Complete ==="

