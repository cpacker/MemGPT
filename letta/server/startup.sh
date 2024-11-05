#!/bin/sh
echo "Starting Letta server at http://localhost:8283"
alembic upgrade head
if [ "$LETTA_ENVIRONMENT" = "DEVELOPMENT" ] ; then
    echo "Starting in development mode!"
    uvicorn letta.server.rest_api.app:app --reload --reload-dir /letta --host 0.0.0.0 --port 8283
else
    uvicorn letta.server.rest_api.app:app --host 0.0.0.0 --port 8283
fi
