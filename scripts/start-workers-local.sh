#!/bin/bash
# Start Celery workers for local demo

set -e
cd "$(dirname "$0")/.."
source venv/bin/activate
cd backend

export DATABASE_URL="sqlite:///./prod.db"
export REDIS_URL="redis://localhost:6379/0"

echo "🧪 Starting Preprocess Worker..."
celery -A app.core.celery_app worker -Q preprocess --loglevel=info --detach --pidfile="/tmp/celery-preprocess.pid" --logfile="celery-preprocess.log"

echo "🧠 Starting Inference Worker..."
celery -A app.core.celery_app worker -Q inference --loglevel=info --detach --pidfile="/tmp/celery-inference.pid" --logfile="celery-inference.log"

echo "🔔 Starting Default Worker..."
celery -A app.core.celery_app worker -Q default --loglevel=info --detach --pidfile="/tmp/celery-default.pid" --logfile="celery-default.log"

echo "✅ Workers started in background."
