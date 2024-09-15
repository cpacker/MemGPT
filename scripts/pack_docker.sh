export MEMGPT_VERSION=$(letta version)
docker buildx build --platform=linux/amd64,linux/arm64,linux/x86_64 --build-arg MEMGPT_ENVIRONMENT=RELEASE -t letta/letta-server:${MEMGPT_VERSION} .
docker push letta/letta-server:${MEMGPT_VERSION}
