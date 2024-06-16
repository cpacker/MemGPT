# The builder image, used to build the virtual environment
FROM python:3.12.2-bookworm as builder
ARG MEMGPT_ENVIRONMENT=PRODUCTION
ENV MEMGPT_ENVIRONMENT=${MEMGPT_ENVIRONMENT}
RUN pip install poetry==1.8.2

ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_VIRTUALENVS_CREATE=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

WORKDIR /app

COPY pyproject.toml poetry.lock ./
RUN poetry lock --no-update
RUN if [ "$MEMGPT_ENVIRONMENT" = "DEVELOPMENT"  ] ; then \
    poetry install --no-root -E "postgres server dev autogen" ; \
    else \
    poetry install --no-root -E "postgres server" && \
    rm -rf $POETRY_CACHE_DIR ;  \
    fi


# The runtime image, used to just run the code provided its virtual environment
FROM python:3.12.2-slim-bookworm as runtime
ARG MEMGPT_ENVIRONMENT=PRODUCTION
ENV MEMGPT_ENVIRONMENT=${MEMGPT_ENVIRONMENT}
ENV VIRTUAL_ENV=/app/.venv \
    PATH="/app/.venv/bin:$PATH"

COPY --from=builder ${VIRTUAL_ENV} ${VIRTUAL_ENV}

COPY ./memgpt /memgpt

EXPOSE 8083

CMD ./memgpt/server/startup.sh

# allow for in-container development and testing
FROM builder as development
ARG MEMGPT_ENVIRONMENT=PRODUCTION
ENV MEMGPT_ENVIRONMENT=${MEMGPT_ENVIRONMENT}
ENV VIRTUAL_ENV=/app/.venv \
    PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH=/
WORKDIR /
COPY ./tests /tests
COPY ./memgpt /memgpt
COPY ./configs/server_config.yaml /root/.memgpt/config
EXPOSE 8083

CMD ./memgpt/server/startup.sh
