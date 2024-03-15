# The builder image, used to build the virtual environment
FROM python:3.11-bookworm as builder

RUN pip install poetry==1.4.2

ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_VIRTUALENVS_CREATE=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

WORKDIR /app

COPY pyproject.toml poetry.lock ./

RUN poetry add psycopg2-binary

RUN poetry install --without dev --no-root -E "postgres server" && rm -rf $POETRY_CACHE_DIR

# The runtime image, used to just run the code provided its virtual environment
FROM python:3.11-slim-bookworm as runtime

ENV VIRTUAL_ENV=/app/.venv \
    PATH="/app/.venv/bin:$PATH"

COPY --from=builder ${VIRTUAL_ENV} ${VIRTUAL_ENV}

COPY memgpt ./memgpt

EXPOSE 8083

ENTRYPOINT ["python", "-m", "uvicorn", "memgpt.server.rest_api.server:app", "--host", "0.0.0.0", "--port", "8083"]
