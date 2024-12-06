#!/bin/sh
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8283}"
echo "Starting Letta server at http://$HOST:$PORT"
alembic upgrade head
exec letta server --host $HOST --port $PORT
