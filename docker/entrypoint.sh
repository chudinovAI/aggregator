#!/bin/bash
set -e

# Run migrations if enabled
if [ "$RUN_MIGRATIONS" = "true" ]; then
    echo "Running database migrations..."
    # Check if alembic exists
    if command -v alembic &> /dev/null; then
        alembic upgrade head
    else
        echo "Alembic not found, skipping migrations."
    fi
fi

# Execute the main command
exec "$@"
