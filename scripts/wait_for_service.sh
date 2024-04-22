#!/bin/bash
# wait-for-it.sh

set -e

host="$1"
shift
cmd="$@"

until curl -s "$host" > /dev/null; do
  >&2 echo "Service is unavailable - sleeping"
  sleep 1
done

>&2 echo "Service is up - executing command"
exec $cmd
