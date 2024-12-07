#!/bin/sh
set -e  # Exit on any error

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8283}"
echo "Attempting to migrate database..."
if ! alembic upgrade head; then
    echo "ERROR: Database migration failed!"
    echo "For more assistance on Letta database migrations, please report an issue on GitHub or reach #support on Discord (https://discord.gg/letta)."
    exit 1
fi

# If ADE is enabled, add the --ade flag to the command
CMD="letta server --host $HOST --port $PORT"
if [ "${ENABLE_ADE:-false}" = "true" ]; then
    CMD="$CMD --ade"
fi

echo "Starting Letta server at http://$HOST:$PORT..."
echo "$CMD"
exec $CMD
