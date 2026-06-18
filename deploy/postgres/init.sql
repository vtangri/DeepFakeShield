-- Initialize DeepFakeShield Database
-- This script runs once when the postgres container is first created

-- Create extensions if needed
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- The deepfakeshield database and user are created by environment variables 
-- in docker-compose.vps.yml, but we can add additional setup here.

-- Ensure the search path is correct
SET search_path TO public;

-- Placeholder for any initial seed data that must exist before migrations
-- (Usually handled by alembic or a python script, but SQL is faster for basics)
