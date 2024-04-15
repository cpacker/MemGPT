export MEMGPT_VERSION=$(memgpt version)
docker buildx build --platform=linux/amd64,linux/arm64,linux/x86_64 --build-arg MEMGPT_ENVIRONMENT=RELEASE -t memgpt/memgpt-server:${MEMGPT_VERSION} .
docker push memgpt/memgpt-server:${MEMGPT_VERSION}
