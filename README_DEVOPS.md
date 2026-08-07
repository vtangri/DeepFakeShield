# 🚀 DeepFakeShield — DevOps & Deployment Guide

This repository has been enhanced with a production-ready DevOps suite for fast, secure, and scalable deployment on a VPS.

## 📁 New Deployment Files

| File | Purpose |
|------|---------|
| `backend/Dockerfile.vps` | Optimized multi-stage Docker image (~50% smaller than dev image). |
| `docker-compose.vps.yml` | Full production stack with resource limits, health checks, and logging. |
| `deploy/nginx/nginx.vps.conf` | Hardened Nginx config with rate limiting, Gzip, and security headers. |
| `deploy/postgres/init.sql` | Database initialization script for production. |
| `scripts/deploy-vps.sh` | One-click deployment script to automate the entire process. |

## 🚀 Deployment to VPS (One-Click)

1. **SSH into your VPS** and clone the repo:
   ```bash
   git clone https://github.com/vtangri/DeepFakeShield.git
   cd DeepFakeShield
   ```

2. **Run the deployment script**:
   ```bash
   bash scripts/deploy-vps.sh
   ```
   *This will automatically generate secure secrets, build Docker images, start all services, and run database migrations.*

## 🖥️ Local Development (Mac)

Since the project uses heavy ML libraries, follow these steps in your local terminal (where you have internet access):

1. **Setup Environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r backend/requirements.txt
   ```

2. **Start Infrastructure (Optional Docker)**:
   ```bash
   docker-compose up -d postgres redis
   ```

3. **Run Backend**:
   ```bash
   cd backend
   alembic upgrade head
   uvicorn app.main:app --reload
   ```

## 📊 Monitoring & Operations

- **API Documentation**: Access at `http://YOUR_IP/docs`
- **Celery Monitoring (Flower)**: 
  1. Open SSH tunnel: `ssh -L 5555:localhost:5555 user@YOUR_IP`
  2. Open browser: `http://localhost:5555` (Login: `admin` / Password: See `.env.prod`)
- **Logs**: `docker compose -f docker-compose.vps.yml logs -f`
- **Backup DB**: `docker exec dfs-postgres pg_dump -U deepfakeshield deepfakeshield > backup.sql`

---
