# AWS Security Group Configuration for Neo4j

## Problem
If you're getting "This site can't be reached" when trying to access Neo4j from outside the server, it's likely because the AWS Security Group is blocking the ports.

## Solution: Configure AWS Security Group

### Step 1: Find Your EC2 Instance
1. Go to AWS Console → EC2 → Instances
2. Find your instance (IP: 54.196.0.75)
3. Note the Security Group name (e.g., `sg-xxxxxxxxx`)

### Step 2: Edit Inbound Rules
1. Click on the Security Group
2. Go to "Inbound rules" tab
3. Click "Edit inbound rules"
4. Click "Add rule" for each port:

**Rule 1: Neo4j HTTP (Browser)**
- Type: Custom TCP
- Port range: `7474`
- Source: 
  - **For your IP only (recommended):** `Your IP Address/32`
  - **For anywhere (less secure):** `0.0.0.0/0`
- Description: `Neo4j Browser HTTP`

**Rule 2: Neo4j Bolt (Database)**
- Type: Custom TCP
- Port range: `7687`
- Source: Same as above
- Description: `Neo4j Bolt Protocol`

### Step 3: Save Rules
Click "Save rules"

## Alternative: Use AWS CLI

If you have AWS CLI configured:

```bash
# Get your security group ID
INSTANCE_ID=$(curl -s http://169.254.169.254/latest/meta-data/instance-id)
SG_ID=$(aws ec2 describe-instances --instance-ids $INSTANCE_ID --query 'Reservations[0].Instances[0].SecurityGroups[0].GroupId' --output text)

# Get your current IP
MY_IP=$(curl -s ifconfig.me)

# Add rule for Neo4j HTTP
aws ec2 authorize-security-group-ingress \
    --group-id $SG_ID \
    --protocol tcp \
    --port 7474 \
    --cidr ${MY_IP}/32 \
    --description "Neo4j Browser from my IP"

# Add rule for Neo4j Bolt
aws ec2 authorize-security-group-ingress \
    --group-id $SG_ID \
    --protocol tcp \
    --port 7687 \
    --cidr ${MY_IP}/32 \
    --description "Neo4j Bolt from my IP"
```

## Verify Configuration

After updating the security group, test from your local machine:

```bash
# Test HTTP port
curl http://54.196.0.75:7474

# Should return JSON response
```

## Security Best Practices

1. **Restrict to Your IP**: Only allow your current IP address (`/32` CIDR)
2. **Use SSH Tunnel**: Even more secure - tunnel through SSH (see EXTERNAL_ACCESS.md)
3. **Change Default Password**: The password `amodels123` should be changed
4. **Use VPN**: Require VPN connection before accessing Neo4j
5. **Enable SSL/TLS**: Configure Neo4j with SSL certificates

## Quick SSH Tunnel Alternative

If you don't want to open ports in the security group, use SSH tunneling:

```bash
ssh -L 7474:localhost:7474 -L 7687:localhost:7687 YOUR_USERNAME@54.196.0.75
```

Then access: `http://localhost:7474`

## Troubleshooting

### Check if ports are open from server
```bash
# On the server
sudo netstat -tlnp | grep 7474
# Should show: 0.0.0.0:7474
```

### Test from inside server
```bash
# On the server
curl http://localhost:7474
# Should return JSON
```

### Check AWS Security Group
```bash
# Get security group ID
INSTANCE_ID=$(curl -s http://169.254.169.254/latest/meta-data/instance-id)
aws ec2 describe-instances --instance-ids $INSTANCE_ID --query 'Reservations[0].Instances[0].SecurityGroups'
```

