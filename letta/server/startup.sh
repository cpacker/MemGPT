#!/bin/sh
set -e  # Exit on any error

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8283}"

# Check if we're configured for Postgres
if [ -n "$LETTA_PG_URI" ]; then
    echo "Postgres configuration detected, attempting to migrate database..."
    if ! alembic upgrade head; then
        echo "ERROR: Database migration failed!"
        echo "Please check your database connection and try again."
        echo "If the problem persists, check the logs for more details."
        exit 1
    fi
    echo "Database migration completed successfully."
else
    echo "No Postgres configuration detected, using SQLite..."
fi

# If ADE is enabled, add the --ade flag to the command
CMD="letta server --host $HOST --port $PORT"
if [ "${ENABLE_ADE:-false}" = "true" ]; then
    CMD="$CMD --ade"
fi

echo "Starting Letta server at http://$HOST:$PORT..."
echo "Executing: $CMD"
exec $CMD
