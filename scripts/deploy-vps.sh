#!/bin/bash
# VPS Deployment Script for DeepFakeShield AI
set -e

echo "🛡️ DeepFakeShield AI - VPS Deployment"
echo "======================================"

# Determine base directory
BASE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$BASE_DIR"

COMPOSE_FILE="docker-compose.vps.yml"

# Check if .env.prod exists
if [ ! -f ".env.prod" ]; then
    echo "⚠️  .env.prod not found! Generating one with secure defaults..."
    DB_PASSWORD=$(openssl rand -hex 16)
    SECRET_KEY=$(openssl rand -hex 32)
    FLOWER_PASSWORD=$(openssl rand -hex 12)
    
    cat > .env.prod << EOF
# Production Environment Variables
DB_PASSWORD=$DB_PASSWORD
SECRET_KEY=$SECRET_KEY
APP_VERSION=1.0.0

# Celery monitoring
FLOWER_USER=admin
FLOWER_PASSWORD=$FLOWER_PASSWORD

# Optional
# OPENAI_API_KEY=
EOF
    echo "✅ Created .env.prod with generated secrets."
fi

# Load environment variables
export $(grep -v '^#' .env.prod | xargs)

# Create necessary directories
echo "📁 Creating production directories..."
mkdir -p deploy/ssl storage/uploads storage/thumbnails storage/reports deploy/postgres

# Generate self-signed SSL if certificates are missing (to allow Nginx to start)
if [ ! -f "deploy/ssl/cert.pem" ]; then
    echo "🔒 Generating temporary self-signed SSL certificates..."
    openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
        -keyout deploy/ssl/key.pem \
        -out deploy/ssl/cert.pem \
        -subj "/C=US/ST=State/L=City/O=DeepFakeShield/OU=AI/CN=localhost"
fi

# Build and start services
echo "🔨 Building and starting Docker containers..."
docker compose -f $COMPOSE_FILE build --pull
docker compose -f $COMPOSE_FILE up -d

# Wait for database to be ready
echo "⏳ Waiting for database to stabilize..."
sleep 10

# Run database migrations
echo "📊 Running database migrations..."
docker compose -f $COMPOSE_FILE exec -T backend alembic upgrade head

echo ""
echo "======================================"
echo "✅ Deployment complete!"
echo ""
echo "Access points:"
echo " - Frontend: http://YOUR_VPS_IP"
echo " - API Docs: http://YOUR_VPS_IP/docs"
echo " - Flower: http://localhost:5555 (after SSH tunneling: ssh -L 5555:localhost:5555 user@YOUR_VPS_IP)"
echo "======================================"
