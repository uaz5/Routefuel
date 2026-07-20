# RouteFuel: Production Deployment & Architecture Guide

## System Overview

RouteFuel is a production-ready AI Gateway that provides:

1. **Live LLM Connectors**: Real HTTP calls to OpenAI and Anthropic
2. **Precision Token Counting**: Tiktoken-based accuracy for cost auditing
3. **Postgres Audit Trail**: Non-blocking async logging of every request
4. **Sub-10ms Routing**: Intelligent model selection with minimal overhead
5. **Cost Tracking**: Automatic ROI calculation and savings reports

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                     Client Request                          │
│            (OpenAI-compatible /v1/chat/completions)         │
└────────────────────────────┬────────────────────────────────┘
                             │
                      ┌──────▼──────┐
                      │Rate Limiter │  (tier-based quotas)
                      └──────┬──────┘
                             │
                    ┌────────▼────────┐
                    │Token Counter    │  (tiktoken-rs)
                    │ Input Tokens    │  Count before sending
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │Route Engine     │  <10ms target
                    │ Scoring         │  Cost/Latency/Quality
                    │ Algorithm       │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │Circuit Breaker  │
                    │ Health Check    │  Fail fast on 5xx
                    └────────┬────────┘
                             │
            ┌────────────────┼────────────────┐
            │                │                │
     ┌──────▼──────┐  ┌─────▼─────┐  ┌──────▼──────┐
     │   OpenAI    │  │ Anthropic │  │  DeepSeek   │
     │   Connector │  │ Connector │  │ (planned)   │
     └──────┬──────┘  └─────┬─────┘  └──────┬──────┘
            │                │                │
            └────────────────┼────────────────┘
                             │
                    ┌────────▼────────┐
                    │Token Counter    │  (tiktoken-rs)
                    │ Output Tokens   │  Verify response
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │Cost Calculator  │
                    │ Precise pricing │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │Async Logger     │  tokio::spawn
                    │ (non-blocking)  │  Write to Postgres
                    └────────┬────────┘
                             │
┌────────────────────────────▼────────────────────────────────┐
│                    Response to Client                       │
│                  + Cost Analysis Metadata                   │
└────────────────────────────────────────────────────────────┘
```

## Deployment Checklist

### Prerequisites
- [ ] Rust 1.70+ installed
- [ ] PostgreSQL 14+ available
- [ ] OpenAI API key obtained
- [ ] Anthropic API key obtained
- [ ] Docker installed (optional)

### Local Development

```bash
# 1. Clone repository
git clone https://github.com/routefuel/routefuel.git
cd routefuel-final

# 2. Setup database
createdb routefuel
echo "DATABASE_URL=postgresql://localhost/routefuel" >> .env

# 3. Run migrations
sqlx database create
sqlx migrate run

# 4. Add API keys
echo "OPENAI_API_KEY=sk_..." >> .env
echo "ANTHROPIC_API_KEY=sk-ant-..." >> .env

# 5. Build
cargo build --release

# 6. Run
./target/release/routefuel
```

### Production Linux Deployment

```bash
# 1. Create user and directories
sudo useradd -m -s /bin/false routefuel
sudo mkdir -p /opt/routefuel
sudo chown routefuel:routefuel /opt/routefuel

# 2. Build statically
RUSTFLAGS="-C target-feature=+crt-static" \
  cargo build --release --target x86_64-unknown-linux-gnu

# 3. Copy binary
sudo cp target/release/routefuel /opt/routefuel/
sudo chown routefuel:routefuel /opt/routefuel/routefuel

# 4. Create .env in /opt/routefuel/.env
sudo tee /opt/routefuel/.env > /dev/null << 'EOF'
OPENAI_API_KEY=sk_...
ANTHROPIC_API_KEY=sk-ant-...
DATABASE_URL=postgresql://routefuel:password@localhost/routefuel
HOST=0.0.0.0
PORT=3000
RUST_LOG=info,routefuel=debug
EOF

sudo chown routefuel:routefuel /opt/routefuel/.env
sudo chmod 600 /opt/routefuel/.env

# 5. Create systemd service
sudo tee /etc/systemd/system/routefuel.service > /dev/null << 'EOF'
[Unit]
Description=RouteFuel LLM Gateway
After=network.target postgresql.service
Wants=postgresql.service

[Service]
Type=simple
User=routefuel
WorkingDirectory=/opt/routefuel
ExecStart=/opt/routefuel/routefuel
Restart=on-failure
RestartSec=10
StandardOutput=journal
StandardError=journal
SyslogIdentifier=routefuel

[Install]
WantedBy=multi-user.target
EOF

# 6. Enable and start
sudo systemctl daemon-reload
sudo systemctl enable routefuel
sudo systemctl start routefuel
sudo systemctl status routefuel
```

### Docker Deployment

```bash
# 1. Build image
docker build -t routefuel:latest .

# 2. Create network
docker network create routefuel

# 3. Run PostgreSQL
docker run -d \
  --name routefuel-postgres \
  --network routefuel \
  -e POSTGRES_DB=routefuel \
  -e POSTGRES_PASSWORD=password \
  -v postgres_data:/var/lib/postgresql/data \
  postgres:15

# 4. Run RouteFuel
docker run -d \
  --name routefuel \
  --network routefuel \
  -p 3000:3000 \
  -e OPENAI_API_KEY=sk_... \
  -e ANTHROPIC_API_KEY=sk-ant-... \
  -e DATABASE_URL=postgresql://postgres:password@routefuel-postgres:5432/routefuel \
  -e RUST_LOG=info \
  routefuel:latest

# 5. Check logs
docker logs -f routefuel
```

## Configuration

### Environment Variables
- `OPENAI_API_KEY`: Required, OpenAI API key
- `ANTHROPIC_API_KEY`: Required, Anthropic API key  
- `DATABASE_URL`: Required, PostgreSQL connection string
- `HOST`: Default `0.0.0.0`, bind address
- `PORT`: Default `3000`, server port
- `RUST_LOG`: Default `info`, logging level

### Database Schema

The migrations automatically create:
- `request_logs` table with 7 indexes
- Columns for tokens, costs, latency, timestamps
- Indexes for fast query performance

## API Endpoints

### POST /v1/chat/completions
OpenAI-compatible chat completion endpoint.

**Request**
```json
{
  "model": "gpt-4o",
  "messages": [
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "Hello!"}
  ],
  "temperature": 0.7,
  "max_tokens": 100
}
```

**Response**
```json
{
  "id": "chatcmpl-123",
  "object": "chat.completion",
  "created": 1705331245,
  "model": "gpt-4o",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Hello! How can I help?"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 30,
    "completion_tokens": 10,
    "total_tokens": 40
  }
}
```

### GET /health
Health check endpoint.

**Response**
```json
{
  "status": "healthy",
  "version": "0.3.0",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

### GET /audit/daily?date=YYYY-MM-DD
Daily audit report with cost savings.

**Response**
```json
{
  "date": "2024-01-15",
  "total_requests": 1250,
  "successful_requests": 1245,
  "failed_requests": 5,
  "total_tokens_in": 187500,
  "total_tokens_out": 625000,
  "total_cost_cents": 7500,
  "total_cost_saved_cents": 287500,
  "avg_latency_ms": 245,
  "providers": [...]
}
```

## Monitoring

### Log Files
```bash
# Follow real-time logs
journalctl -u routefuel -f

# Filter by level
journalctl -u routefuel -p err

# Last 100 lines
journalctl -u routefuel -n 100
```

### Key Metrics to Monitor

1. **Latency**: `routing_decision_ms` should be <10ms
2. **Error Rate**: Monitor `status != 'success'` in request_logs
3. **Cost Savings**: Daily `cost_saved_cents` aggregation
4. **Circuit Breaker**: Check if providers are isolated

### PostgreSQL Queries

```sql
-- Daily costs
SELECT DATE(created_at), SUM(cost_cents), SUM(cost_saved_cents)
FROM request_logs
WHERE status = 'success'
GROUP BY DATE(created_at)
ORDER BY DATE(created_at) DESC;

-- Provider performance
SELECT provider, COUNT(*), AVG(latency_ms), SUM(cost_saved_cents)
FROM request_logs
WHERE status = 'success'
GROUP BY provider;

-- Routing latency
SELECT AVG(routing_decision_ms), MAX(routing_decision_ms), MIN(routing_decision_ms)
FROM request_logs;
```

## Performance Tuning

### Database

```bash
# Increase connection pool
# In src/main.rs:
// pool = sqlx::postgres::PgPoolOptions::new()
//    .max_connections(50)

# Create indexes before high traffic
CREATE INDEX CONCURRENTLY idx_request_logs_created_at 
ON request_logs(created_at);

# Analyze for query optimizer
ANALYZE request_logs;
```

### Server

```bash
# Increase file descriptors
ulimit -n 65536

# Tune TCP settings
sudo sysctl -w net.core.somaxconn=65536
sudo sysctl -w net.ipv4.tcp_max_syn_backlog=65536
```

## Troubleshooting

### Database Connection Failed
```bash
# Check PostgreSQL is running
sudo systemctl status postgresql

# Test connection
psql $DATABASE_URL -c "SELECT 1;"

# Check migrations
sqlx migrate list
```

### Circuit Breaker Open
```bash
# Check API keys
echo $OPENAI_API_KEY | head -c 10
echo $ANTHROPIC_API_KEY | head -c 15

# Check network connectivity
curl https://api.openai.com/v1/models -H "Authorization: Bearer $OPENAI_API_KEY" | jq .

# Restart service
sudo systemctl restart routefuel
```

### Token Count Mismatch
```bash
# Enable debug logging
RUST_LOG=routefuel=debug cargo run

# Look for verification messages
grep "Verified output tokens" /var/log/routefuel.log
```

## Scaling

### Horizontal Scaling
1. Deploy multiple RouteFuel instances
2. Use load balancer (nginx, AWS ALB)
3. Share PostgreSQL database
4. Use connection pooling (PgBouncer)

### Vertical Scaling
- Increase `max_connections` in database pool
- Tune Linux kernel parameters
- Use larger instance type

## Security

### Production Hardening
- [ ] Use HTTPS/TLS (behind nginx/ALB)
- [ ] Implement API key authentication
- [ ] Use VPC security groups
- [ ] Enable PostgreSQL SSL connections
- [ ] Use secrets management (Vault, AWS Secrets)
- [ ] Enable audit logging
- [ ] Monitor access logs

### Database
```sql
-- Create dedicated user
CREATE USER routefuel WITH PASSWORD 'secure_password';
GRANT CONNECT ON DATABASE routefuel TO routefuel;
GRANT USAGE ON SCHEMA public TO routefuel;
GRANT ALL ON ALL TABLES IN SCHEMA public TO routefuel;
```

## Backup & Recovery

### PostgreSQL Backup
```bash
# Daily backup
pg_dump $DATABASE_URL | gzip > /backups/routefuel-$(date +%Y%m%d).sql.gz

# Restore
gunzip < /backups/routefuel-20240115.sql.gz | psql $DATABASE_URL
```

### Binary Backup
```bash
# Copy binary to safe location
cp /opt/routefuel/routefuel /backups/routefuel-$(date +%Y%m%d)
```

## Support

- **Issues**: GitHub Issues
- **Documentation**: /docs directory
- **Email**: support@routefuel.io

---

**Version**: 0.3.0  
**Last Updated**: January 2024
