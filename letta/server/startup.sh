#!/bin/sh
PORT="${PORT:-8283}"
echo "Starting Letta server at http://localhost:$PORT"

# Check if LETTA_PG_URI or LETTA_PG_DB is set and run alembic upgrade if either is
if [ -n "$LETTA_PG_URI" ] || [ -n "$LETTA_PG_DB" ]; then
    echo "LETTA_PG_URI or LETTA_PG_DB is set, running alembic upgrade head"
    alembic upgrade head
fi

if [ "$MEMGPT_ENVIRONMENT" = "DEVELOPMENT" ]; then
    echo "Starting in development mode!"
    uvicorn letta.server.rest_api.app:app --reload --reload-dir /letta --host 0.0.0.0 --port $PORT
else
    # Production start command here (replace with the actual production command)
    echo "Starting in production mode!"
    uvicorn letta.server.rest_api.app:app --host 0.0.0.0 --port $PORT
fi
