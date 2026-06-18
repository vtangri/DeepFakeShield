#!/bin/bash
# Production Deployment Script for DeepFakeShield AI

set -e

echo "🛡️ DeepFakeShield AI - Production Deployment"
echo "============================================="

cd "$(dirname "$0")/.."

# Detect OS
OS_NAME=$(uname)
COMPOSE_FILE="docker-compose.prod.yml"

if [[ "$OS_NAME" == "Darwin" ]]; then
    echo "🍎 Mac OS detected. Switching to CPU optimization..."
    COMPOSE_FILE="docker-compose.prod.mac.yml"
fi

echo "📄 Using compose file: $COMPOSE_FILE"

# Check environment or create auto-generated one
if [ ! -f ".env.prod" ]; then
    echo "⚠️  .env.prod not found. Generating secure configuration..."
    
    # Generate secrets
    DB_PASSWORD=$(openssl rand -hex 16)
    SECRET_KEY=$(openssl rand -hex 32)
    
    cat > .env.prod << EOF
# Production Environment Variables (Auto-Generated)
DB_PASSWORD=$DB_PASSWORD
SECRET_KEY=$SECRET_KEY

# Optional: External services
# SENTRY_DSN=
# OPENAI_API_KEY=
EOF
    echo "✅ Created .env.prod with generated secrets."
fi

# Load environment
export $(grep -v '^#' .env.prod | xargs)

# Create directories
echo "📁 Creating directories..."
mkdir -p deploy/ssl storage models

# Generate Self-Signed SSL if missing (prevents Nginx start failure)
if [ ! -f "deploy/ssl/cert.pem" ]; then
    echo "🔒 Generating self-signed SSL certificates..."
    openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
        -keyout deploy/ssl/key.pem \
        -out deploy/ssl/cert.pem \
        -subj "/C=US/ST=State/L=City/O=DeepFakeShield/OU=AI/CN=localhost"
fi

# Build images
echo "🔨 Building Docker images..."
docker compose -f $COMPOSE_FILE build

# Start services
echo "🚀 Starting services..."
docker compose -f $COMPOSE_FILE up -d

# Wait for database
echo "⏳ Waiting for database..."
sleep 15

# Run migrations
echo "📊 Running database migrations..."
docker compose -f $COMPOSE_FILE exec -T backend alembic upgrade head

# Status
echo ""
echo "============================================="
echo "✅ Deployment complete!"
echo ""
echo "Services:"
echo "  - Frontend: http://localhost"
echo "  - API: http://localhost/api"
echo "  - API Docs: http://localhost/docs"
echo "  - Flower: http://localhost:5555"
echo ""
echo "Commands:"
echo "  - View logs: docker compose -f $COMPOSE_FILE logs -f"
echo "  - Stop: docker compose -f $COMPOSE_FILE down"
echo "============================================="
