#!/bin/sh
echo "Generating OpenAPI schema..."

# check if poetry is installed
if ! command -v poetry &> /dev/null
then
    echo "Poetry could not be found. Please install poetry to generate the OpenAPI schema."
    exit
fi

# generate OpenAPI schema
poetry run python -c 'from letta.server.rest_api.app import app, generate_openapi_schema; generate_openapi_schema(app);'
