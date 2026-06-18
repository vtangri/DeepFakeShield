#!/bin/bash
# DeepFakeShield AI - Development Setup Script

set -e

echo "🛡️ DeepFakeShield AI - Development Setup"
echo "=========================================="

# Change to project directory
cd "$(dirname "$0")/.."

# Check for Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.9+ first."
    exit 1
fi

# Check for Docker
if ! command -v docker &> /dev/null; then
    echo "⚠️  Docker is not installed. You can run without Docker but need PostgreSQL and Redis."
fi

# Create virtual environment
echo ""
echo "📦 Setting up Python virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi
source venv/bin/activate

# Install dependencies
echo ""
echo "📥 Installing Python dependencies..."
pip install --upgrade pip
pip install -r backend/requirements.txt

# Create .env if not exists
if [ ! -f ".env" ]; then
    echo ""
    echo "📝 Creating .env file from example..."
    cp .env.example .env
    echo "⚠️  Please edit .env and set your configuration values!"
fi

# Create storage directory
echo ""
echo "📁 Creating storage directories..."
mkdir -p storage/uploads storage/reports storage/temp

# Run with Docker or locally
echo ""
echo "=========================================="
echo "✅ Setup complete!"
echo ""
echo "To run with Docker:"
echo "  docker-compose up -d"
echo ""
echo "To run locally (requires PostgreSQL and Redis):"
echo "  source venv/bin/activate"
echo "  cd backend"
echo "  uvicorn app.main:app --reload --host 0.0.0.0 --port 8000"
echo ""
echo "Then open: http://localhost:8000"
echo "=========================================="
