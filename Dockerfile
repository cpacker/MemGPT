# Start with pgvector base for builder
FROM ankane/pgvector:v0.5.1 AS builder

# Install Python and required packages
RUN apt-get update && apt-get install -y \
    python3 \
    python3-venv \
    python3-pip \
    python3-full \
    build-essential \
    libpq-dev \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

ARG LETTA_ENVIRONMENT=PRODUCTION
ENV LETTA_ENVIRONMENT=${LETTA_ENVIRONMENT} \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_VIRTUALENVS_CREATE=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

WORKDIR /app

# Create and activate virtual environment
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Now install poetry in the virtual environment
RUN pip install --no-cache-dir poetry==1.8.2

# Copy dependency files first
COPY pyproject.toml poetry.lock ./
# Then copy the rest of the application code
COPY . .

RUN poetry lock --no-update && \
    poetry install --all-extras && \
    rm -rf $POETRY_CACHE_DIR

# Runtime stage
FROM ankane/pgvector:v0.5.1 AS runtime

# Install Python packages
RUN apt-get update && apt-get install -y \
    python3 \
    python3-venv \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /app

ARG LETTA_ENVIRONMENT=PRODUCTION
ENV LETTA_ENVIRONMENT=${LETTA_ENVIRONMENT} \
    VIRTUAL_ENV="/app/.venv" \
    PATH="/app/.venv/bin:$PATH" \
    POSTGRES_USER=letta \
    POSTGRES_PASSWORD=letta \
    POSTGRES_DB=letta

WORKDIR /app

# Copy virtual environment and app from builder
COPY --from=builder /app .

# Copy initialization SQL if it exists
COPY init.sql /docker-entrypoint-initdb.d/

EXPOSE 8283 5432

ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
CMD ["./letta/server/startup.sh"]
