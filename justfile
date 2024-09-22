set dotenv-load

DOCKER_REGISTRY := "${REGION}-docker.pkg.dev/${PROJECT_NAME}/${REGISTRY_NAME}"
HELM_CHARTS_DIR := "helm"
HELM_CHART_NAME := "memgpt-server"
TAG := env_var_or_default("TAG", "latest")

# List all Justfile commands
@list:
    echo "🚧 Listing Justfile commands..."
    just --list

# Authenticate with GCP
authenticate:
    @echo "🔐 Authenticating with Google Cloud..."
    gcloud auth application-default login --project ${PROJECT_NAME}
    @echo "🔐 Configuring Docker authentication..."
    gcloud auth configure-docker ${REGION}-docker.pkg.dev --quiet

# Configure kubectl
configure-kubectl:
    @echo "🔧 Configuring kubectl for the Letta cluster..."
    gcloud container clusters get-credentials letta --region ${REGION} --project ${PROJECT_NAME}

# Build the Docker images
build:
    @echo "🚧 Building multi-architecture Docker images with tag: {{TAG}}..."
    docker buildx create --use
    docker buildx build --progress=plain --platform linux/amd64 -t {{DOCKER_REGISTRY}}/memgpt-server:{{TAG}} . --load

# Push the Docker images to the registry
push:
    @echo "🚀 Pushing Docker images to registry with tag: {{TAG}}..."
    docker push {{DOCKER_REGISTRY}}/memgpt-server:{{TAG}}

# Deploy the Helm chart
deploy: push
    @echo "🚧 Deploying Helm chart..."
    helm upgrade --install {{HELM_CHART_NAME}} {{HELM_CHARTS_DIR}}/{{HELM_CHART_NAME}} \
        --set image.repository={{DOCKER_REGISTRY}}/memgpt-server \
        --set image.tag={{TAG}} \
        --set secrets.OPENAI_API_KEY=${OPENAI_API_KEY} \
        --set secrets.MEMGPT_SERVER_PASS=${MEMGPT_SERVER_PASS} \
        --set secrets.MEMGPT_PG_DB=${MEMGPT_PG_DB} \
        --set secrets.MEMGPT_PG_USER=${MEMGPT_PG_USER} \
        --set secrets.MEMGPT_PG_PASSWORD=${MEMGPT_PG_PASSWORD} \
        --set secrets.MEMGPT_PG_HOST=${MEMGPT_PG_HOST} \
        --set secrets.POSTGRES_URI=${POSTGRES_URI}

# Destroy the Helm chart
destroy:
    @echo "🚧 Undeploying web service Helm chart..."
    helm uninstall {{HELM_CHART_NAME}}

# Show environment variables on the pod
show-env:
    @echo "🚧 Showing environment variables..."
    kubectl exec -it $(kubectl get pods -l app.kubernetes.io/name=memgpt-server -o jsonpath="{.items[0].metadata.name}") -- env

# Show secret
@show-secret:
    echo "🚧 Showing secret..."
    kubectl get secret memgpt-server-env-secret -o jsonpath='{.data}' | jq -r 'to_entries[] | "\(.key) \(.value | @base64d)"'

# SSH into the pod
ssh:
    kubectl exec -it $(kubectl get pods -l app.kubernetes.io/name=memgpt-server -o jsonpath="{.items[0].metadata.name}") -- /bin/sh

# Get logs
logs:
    kubectl logs $(kubectl get pods -l app.kubernetes.io/name=memgpt-server -o jsonpath="{.items[0].metadata.name}")

# Describe the pod
describe-server:
    kubectl describe pod $(kubectl get pods -l app.kubernetes.io/name=memgpt-server -o jsonpath="{.items[0].metadata.name}")
