FROM python:3.12.2-bookworm AS builder
ARG LETTA_ENVIRONMENT=PRODUCTION
ENV LETTA_ENVIRONMENT=${LETTA_ENVIRONMENT} \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_VIRTUALENVS_CREATE=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

WORKDIR /app

RUN pip install --no-cache-dir poetry==1.8.2

# First copy only dependency files
COPY pyproject.toml poetry.lock ./

# Then copy the rest of the application code
COPY . .

RUN poetry lock --no-update && \
    poetry install --all-extras && \
    rm -rf $POETRY_CACHE_DIR

FROM python:3.12.2-slim-bookworm AS runtime
ARG LETTA_ENVIRONMENT=PRODUCTION
ENV LETTA_ENVIRONMENT=${LETTA_ENVIRONMENT} \
    VIRTUAL_ENV=/app/.venv \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

# Copy the entire app directory but use .dockerignore to exclude unnecessary files
COPY --from=builder /app .

EXPOSE 8283

CMD ["./letta/server/startup.sh"]
