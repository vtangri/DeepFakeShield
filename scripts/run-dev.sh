#!/bin/bash
# DeepFakeShield AI - Development Runner

set -e

cd "$(dirname "$0")/.."

echo "🛡️ Starting DeepFakeShield AI..."

# Check if using Docker
if [ "$1" == "--docker" ] || [ "$1" == "-d" ]; then
    echo "🐳 Starting with Docker Compose..."
    docker-compose up -d
    echo ""
    echo "✅ Services started!"
    echo "   API: http://localhost:8000"
    echo "   Docs: http://localhost:8000/docs"
    echo ""
    echo "To view logs: docker-compose logs -f"
    echo "To stop: docker-compose down"
    exit 0
fi

# Local development
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found. Run ./scripts/setup.sh first."
    exit 1
fi

source venv/bin/activate

echo "🚀 Starting FastAPI server..."
cd backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
