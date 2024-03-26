export MEMGPT_VERSION=$(memgpt version)
#docker build -t memgpt/memgpt-server:${MEMGPT_VERSION} --platform linux/x86_64 .
docker buildx build --platform=linux/amd64,linux/arm64,linux/x86_64 -t memgpt/memgpt-server:${MEMGPT_VERSION} .
docker push memgpt/memgpt-server:${MEMGPT_VERSION}
