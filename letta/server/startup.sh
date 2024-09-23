#!/bin/sh
echo "Starting MEMGPT server..."
if [ "$MEMGPT_ENVIRONMENT" = "DEVELOPMENT" ] ; then
    echo "Starting in development mode!"
    uvicorn letta.server.rest_api.app:app --reload --reload-dir /letta --host 0.0.0.0 --port 8083
else
    uvicorn letta.server.rest_api.app:app --host 0.0.0.0 --port 8083
fi
