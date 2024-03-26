export DOCKER_DEFAULT_PLATFORM=linux/amd64
export MEMGPT_VERSION=$(memgpt version)
docker build -t memgpt/memgpt-server:${MEMGPT_VERSION} .
docker push memgpt/memgpt-server:${MEMGPT_VERSION}
