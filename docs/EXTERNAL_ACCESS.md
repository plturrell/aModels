# External Access to Neo4j Browser

This guide explains how to access Neo4j Browser from outside the server.

## Option 1: SSH Tunnel (Recommended - Most Secure)

This is the safest method as it doesn't expose Neo4j to the internet.

### Setup SSH Tunnel

On your local machine, run:

```bash
ssh -L 7474:localhost:7474 -L 7687:localhost:7687 user@your-server-ip
```

Replace:
- `user` with your SSH username
- `your-server-ip` with your server's IP address or hostname

### Access Neo4j Browser

Once the tunnel is active:
1. Open your browser
2. Navigate to: `http://localhost:7474`
3. Login with:
   - Username: `neo4j`
   - Password: `amodels123`

### Keep Tunnel Running in Background

```bash
# Run in background
ssh -f -N -L 7474:localhost:7474 -L 7687:localhost:7687 user@your-server-ip

# Or with autossh (if installed)
autossh -M 20000 -f -N -L 7474:localhost:7474 -L 7687:localhost:7687 user@your-server-ip
```

## Option 2: Direct External Access (Requires Configuration)

If you want direct access without SSH tunneling, you need to:

### 1. Configure Neo4j to Accept External Connections

Neo4j by default only binds to `localhost`. To allow external access, you need to configure it.

**Option A: Using Environment Variables (if Neo4j is in docker-compose.yml)**

Add to Neo4j service:
```yaml
neo4j:
  environment:
    - NEO4J_AUTH=neo4j/amodels123
    - NEO4J_dbms_default__listen__address=0.0.0.0
    - NEO4J_dbms_connector_http_listen__address=0.0.0.0:7474
    - NEO4J_dbms_connector_bolt_listen__address=0.0.0.0:7687
```

**Option B: Using Neo4j Configuration File**

If Neo4j is running standalone, edit `/conf/neo4j.conf`:
```properties
dbms.default_listen_address=0.0.0.0
dbms.connector.http.listen_address=0.0.0.0:7474
dbms.connector.bolt.listen_address=0.0.0.0:7687
```

### 2. Configure Firewall

Allow ports 7474 and 7687 through your firewall:

```bash
# Ubuntu/Debian (ufw)
sudo ufw allow 7474/tcp
sudo ufw allow 7687/tcp

# Or for specific IP only (more secure)
sudo ufw allow from YOUR_IP_ADDRESS to any port 7474
sudo ufw allow from YOUR_IP_ADDRESS to any port 7687

# CentOS/RHEL (firewalld)
sudo firewall-cmd --permanent --add-port=7474/tcp
sudo firewall-cmd --permanent --add-port=7687/tcp
sudo firewall-cmd --reload

# AWS Security Groups
# Add inbound rules for ports 7474 and 7687 in your EC2 security group
```

### 3. Access Neo4j Browser

1. Open browser
2. Navigate to: `http://YOUR_SERVER_IP:7474`
3. Login with:
   - Username: `neo4j`
   - Password: `amodels123`

### 4. Security Considerations

⚠️ **WARNING**: Exposing Neo4j directly to the internet is a security risk. Consider:

1. **Use Strong Password**: Change the default password
2. **Restrict Access**: Use firewall rules to only allow specific IPs
3. **Use HTTPS**: Configure Neo4j with SSL/TLS certificates
4. **VPN**: Require VPN connection before accessing Neo4j
5. **Rate Limiting**: Configure rate limiting to prevent brute force attacks

## Option 3: Reverse Proxy (Advanced)

Use nginx or another reverse proxy with SSL/TLS:

```nginx
server {
    listen 443 ssl;
    server_name neo4j.yourdomain.com;
    
    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;
    
    location / {
        proxy_pass http://localhost:7474;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

## Finding Your Server IP

```bash
# Public IP
curl ifconfig.me
# or
curl ipinfo.io/ip

# Private IP (if on local network)
hostname -I
```

## Testing Connection

From your local machine:

```bash
# Test HTTP port
curl http://YOUR_SERVER_IP:7474

# Test Bolt port
# (Use a Neo4j client or the browser)
```

## Troubleshooting

### Port Not Accessible

1. **Check if Neo4j is listening on 0.0.0.0:**
   ```bash
   docker exec neo4j netstat -tlnp | grep 7474
   ```

2. **Check firewall status:**
   ```bash
   sudo ufw status
   # or
   sudo firewall-cmd --list-all
   ```

3. **Check if port is open:**
   ```bash
   sudo netstat -tlnp | grep 7474
   ```

4. **Check Docker port mapping:**
   ```bash
   docker port neo4j
   ```

### Connection Refused

- Verify Neo4j is running: `docker ps | grep neo4j`
- Check Neo4j logs: `docker logs neo4j`
- Verify bind address configuration
- Check firewall rules

### Authentication Failed

- Verify password: Check `NEO4J_AUTH` environment variable
- Reset password if needed (see Neo4j documentation)

## Quick Reference

**✅ Direct Access (Now Working):**
```
http://54.196.0.75:7474
Username: neo4j
Password: amodels123
Bolt URL: bolt://ec2-54-196-0-75.compute-1.amazonaws.com:7687
```

**SSH Tunnel (Alternative):**
```bash
ssh -L 7474:localhost:7474 -L 7687:localhost:7687 user@server
# Then access: http://localhost:7474
```

## ✅ Status: External Access Configured

AWS Security Groups have been configured and Neo4j is now accessible via:
- **HTTP**: http://54.196.0.75:7474
- **Bolt**: bolt://ec2-54-196-0-75.compute-1.amazonaws.com:7687

