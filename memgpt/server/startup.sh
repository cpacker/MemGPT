#!/bin/sh
set -e
echo "updating database..."
cd /memgpt && alembic upgrade head
echo "Database updated!"
echo "Starting MEMGPT server..."
cd ../
if [ "$MEMGPT_ENVIRONMENT" = "DEVELOPMENT" ] ; then
    echo "Starting in development mode!"
    uvicorn memgpt.server.rest_api.app:app --reload --reload-dir /memgpt --host 0.0.0.0 --port 8083
else
    uvicorn memgpt.server.rest_api.app:app --host 0.0.0.0 --port 8083
fi
