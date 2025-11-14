-- Initialize Gitea database
-- This script is automatically executed by PostgreSQL on first startup

-- Create Gitea database user
CREATE USER gitea WITH PASSWORD 'gitea_password';

-- Create Gitea database
CREATE DATABASE gitea WITH OWNER gitea ENCODING 'UTF8';

-- Grant all privileges
GRANT ALL PRIVILEGES ON DATABASE gitea TO gitea;

-- Connect to Gitea database and grant schema privileges
\c gitea;
GRANT ALL ON SCHEMA public TO gitea;

-- Log completion
SELECT 'Gitea database initialized successfully' as status;
