#!/bin/bash
# DeepFakeShield AI - Local Production Runner
# Runs the app without Docker using SQLite

set -e

cd "$(dirname "$0")/.."

echo "🛡️ Starting DeepFakeShield AI (Local Live Mode)..."

# Activate venv
if [ -d "venv" ]; then
    source venv/bin/activate
else
    echo "⚠️  venv not found. Attempting to use system python or setup..."
    ./scripts/setup.sh
    source venv/bin/activate
fi

# Set Production Envs for Local Run
export DATABASE_URL="sqlite:///./prod.db"
export SECRET_KEY=$(openssl rand -hex 32)
export DEBUG=false
# Disable GPU if not available to avoid warnings/errors
export ENABLE_GPU=false 

# Create storage dir if missing
mkdir -p backend/storage

echo "📦 Database: SQLite (./prod.db)"
echo "🚀 Starting Server on http://0.0.0.0:8001"

cd backend

# Run Migrations (handling sqlite async issue if needed, but sync engine handles migrations)
echo "📊 Running migrations..."
alembic upgrade head

# Start App
uvicorn app.main:app --host 0.0.0.0 --port 8001 --workers 4
