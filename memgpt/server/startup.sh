#!/bin/sh
echo "Starting MEMGPT server..."
if [ "$MEMGPT_ENVIRONMENT" = "DEVELOPMENT" ] ; then
    uvicorn memgpt.server.rest_api.server:app --reload --reload-dir /memgpt --host 0.0.0.0 --port 8083
else
    uvicorn memgpt.server.rest_api.server:app --host 0.0.0.0 --port 8083
fi
