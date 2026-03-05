#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# deploy.sh — One-command deployment of LLM Broker to Azure Container App
#
# This script is idempotent: running it multiple times will update the
# existing deployment rather than creating duplicates.
#
# Prerequisites:
#   - Azure CLI (az) installed and logged in
#   - Docker installed (optional — falls back to az acr build if unavailable)
#
# Usage:
#   ./infra/deploy.sh
#
# Customisation (environment variables):
#   RESOURCE_GROUP   — Azure resource group name     (default: llm-broker-rg)
#   LOCATION         — Azure region                  (default: westeurope)
#   ENVIRONMENT_NAME — Base name for all resources   (default: llm-broker)
#   IMAGE_TAG        — Docker image tag              (default: latest)
#   ANTHROPIC_API_KEY — Anthropic provider API key   (default: empty)
#   AZURE_API_KEY     — Azure provider API key       (default: empty)
#   OPENAI_API_KEY    — OpenAI provider API key      (default: empty)
# ---------------------------------------------------------------------------
set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration — override via environment variables
# ---------------------------------------------------------------------------
RESOURCE_GROUP="${RESOURCE_GROUP:-llm-broker-rg}"
LOCATION="${LOCATION:-westeurope}"
ENVIRONMENT_NAME="${ENVIRONMENT_NAME:-llm-broker}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
IMAGE_NAME="${ENVIRONMENT_NAME}:${IMAGE_TAG}"

# Provider API keys (pass via env vars; use "placeholder" if unset since
# Azure Container Apps rejects empty secret values)
ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY:-placeholder}"
AZURE_API_KEY="${AZURE_API_KEY:-placeholder}"
OPENAI_API_KEY="${OPENAI_API_KEY:-placeholder}"

# Resolve the project root (one level up from infra/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

echo "============================================="
echo "  LLM Broker — Azure Container App Deploy"
echo "============================================="
echo ""
echo "  Resource Group:   ${RESOURCE_GROUP}"
echo "  Location:         ${LOCATION}"
echo "  Environment Name: ${ENVIRONMENT_NAME}"
echo "  Image:            ${IMAGE_NAME}"
echo ""

# ---------------------------------------------------------------------------
# Step 1: Create resource group (idempotent — no-op if it already exists)
# ---------------------------------------------------------------------------
echo "[1/5] Ensuring resource group '${RESOURCE_GROUP}' exists..."
az group create \
  --name "${RESOURCE_GROUP}" \
  --location "${LOCATION}" \
  --output none

# ---------------------------------------------------------------------------
# Step 2: Deploy the Bicep template to provision ACR + Container App
#         Environment. We do this first so the ACR exists before we push.
#         On the first run the Container App will fail to pull the image
#         (it hasn't been pushed yet), but the ACR is created. On subsequent
#         runs, the image already exists in the ACR.
# ---------------------------------------------------------------------------
echo "[2/5] Deploying Bicep template (ACR + Container App Environment + App)..."
DEPLOY_OUTPUT=$(az deployment group create \
  --resource-group "${RESOURCE_GROUP}" \
  --template-file "${SCRIPT_DIR}/containerapp.bicep" \
  --parameters \
    environmentName="${ENVIRONMENT_NAME}" \
    imageName="${IMAGE_NAME}" \
    anthropicApiKey="${ANTHROPIC_API_KEY}" \
    azureApiKey="${AZURE_API_KEY}" \
    openaiApiKey="${OPENAI_API_KEY}" \
  --query "properties.outputs" \
  --output json)

# Extract ACR login server from deployment outputs
ACR_LOGIN_SERVER=$(echo "${DEPLOY_OUTPUT}" | python3 -c "import sys,json; print(json.load(sys.stdin)['acrLoginServer']['value'])")
APP_URL=$(echo "${DEPLOY_OUTPUT}" | python3 -c "import sys,json; print(json.load(sys.stdin)['appUrl']['value'])")

echo "  ACR: ${ACR_LOGIN_SERVER}"

# ---------------------------------------------------------------------------
# Step 3+4: Build and push the Docker image
#           Use az acr build (cloud build) if local Docker is unavailable.
# ---------------------------------------------------------------------------
FULL_IMAGE="${ACR_LOGIN_SERVER}/${IMAGE_NAME}"
ACR_NAME="${ACR_LOGIN_SERVER%%.*}"

if docker info &>/dev/null; then
  echo "[3/5] Logging in to ACR '${ACR_LOGIN_SERVER}'..."
  az acr login --name "${ACR_NAME}"
  echo "[4/5] Building and pushing Docker image '${FULL_IMAGE}'..."
  docker build -t "${FULL_IMAGE}" "${PROJECT_ROOT}"
  docker push "${FULL_IMAGE}"
else
  echo "[3/5] Local Docker unavailable — using az acr build (cloud build)..."
  echo "[4/5] Building Docker image in ACR '${ACR_NAME}'..."
  az acr build \
    --registry "${ACR_NAME}" \
    --image "${IMAGE_NAME}" \
    "${PROJECT_ROOT}"
fi

# ---------------------------------------------------------------------------
# Step 5: Update the Container App to use the newly pushed image
#         This forces a new revision even if the Bicep template hasn't changed.
# ---------------------------------------------------------------------------
echo "[5/5] Updating Container App to use the latest image..."
az containerapp update \
  --name "${ENVIRONMENT_NAME}" \
  --resource-group "${RESOURCE_GROUP}" \
  --image "${FULL_IMAGE}" \
  --output none

# ---------------------------------------------------------------------------
# Done — print the app URL
# ---------------------------------------------------------------------------
echo ""
echo "============================================="
echo "  Deployment complete!"
echo "============================================="
echo ""
echo "  App URL:  ${APP_URL}"
echo ""
echo "  Health check:"
echo "    curl ${APP_URL}/health"
echo ""
echo "  Example request:"
echo "    curl -X POST ${APP_URL}/v1/chat/completions \\"
echo "      -H 'Content-Type: application/json' \\"
echo "      -H 'Authorization: Bearer sk-internal-xyz789' \\"
echo "      -d '{\"messages\": [{\"role\": \"user\", \"content\": \"Hello\"}]}'"
echo ""
