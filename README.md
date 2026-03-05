<div align="center">

```
 _      _      __  __     ____  ____   ___  _  __ _____ ____
| |    | |    |  \/  |   | __ )|  _ \ / _ \| |/ /| ____|  _ \
| |    | |    | |\/| |   |  _ \| |_) | | | | ' / |  _| | |_) |
| |___ | |___ | |  | |   | |_) |  _ <| |_| | . \ | |___|  _ <
|_____||_____||_|  |_|   |____/|_| \_\\___/|_|\_\|_____|_| \_\
```

</div>

# LLM Broker

**Compliance-aware LLM routing broker with OpenAI-compatible API.**

A single endpoint that transparently routes requests to the right LLM provider based on per-repo compliance rules, data residency requirements, and cost optimization. Developers point their tools (Claude Code, custom apps, CI pipelines) at the broker and forget about compliance -- the broker handles provider selection, PII redaction, and fallback routing automatically.

---

## Architecture

```
Developer Request
  |  POST /v1/chat/completions  (OpenAI format)
  |  Authorization: Bearer sk-fintech-abc123
  |  (optional) X-Size: medium
  v
+-----------------------------------------+
|  Auth: API Key -> Repo Resolution       |
|  (keys.yml maps key to repo config)     |
+--------------------+--------------------+
                     |
                     v
+-----------------------------------------+
|  Layer 1: Compliance Gateway            |
|  - Filter providers by allowlist        |
|  - Filter by data residency            |
|  - Enforce tier ceiling                 |
|  - PII detection & redaction            |
|  Output: sanitized request +            |
|          list of eligible models        |
+--------------------+--------------------+
                     |
                     v
+-----------------------------------------+
|  Layer 2: Intelligent Router            |
|  - RouteLLM: strong or weak? (~1ms)    |
|  - Size hint override from header       |
|  - Group eligible models by tier        |
|  - Pick cheapest in selected tier       |
|  - Build fallback chain                 |
|  Output: ordered model list             |
+--------------------+--------------------+
                     |
                     v
+-----------------------------------------+
|  Layer 3: Provider Proxy (LiteLLM)      |
|  - Dispatch to provider                 |
|  - Stream / tool-call passthrough       |
|  - Auto-retry with fallback             |
|  - Log cost + latency (JSON)            |
+--------------------+--------------------+
                     |
                     v
         Provider (Anthropic / Azure / OpenAI / Ollama)
```

## Quick Start

### Docker Compose

```bash
git clone <repo-url>
cd llm-broker
docker compose up --build

# Health check
curl http://localhost:8000/health
# {"status": "ok", "repos": ["fintech-app", "internal-tooling"]}
```

### Local Development

```bash
pip install -e ".[dev]"
uvicorn llm_broker.main:app --host 0.0.0.0 --port 8000
pytest tests/ -v
```

---

## Configuration

All configuration lives in `configs/` as YAML files.

### Provider Registry (`configs/providers.yml`)

Defines available LLM providers, their models, regions, and pricing:

```yaml
providers:
  anthropic-saas:
    litellm_prefix: "anthropic/"
    region: us
    deployment: saas
    models:
      - id: claude-sonnet
        litellm_model: "anthropic/claude-sonnet-4-20250514"
        tier: premium
        quality: 0.9
        cost_per_1k_tokens: 0.015
```

Each model has:
- **id** -- internal identifier used by the router
- **litellm_model** -- LiteLLM model string for dispatch
- **tier** -- `free`, `standard`, or `premium` (used for tier ceiling enforcement)
- **quality** -- quality score (0-1) for routing decisions
- **cost_per_1k_tokens** -- cost estimate for logging

### Repo Configs (`configs/repos/*.yml`)

Per-repository compliance rules. Each repo gets its own YAML file:

```yaml
repo: fintech-app
allowed_providers:
  - azure-foundry          # Only Azure (EU-hosted) allowed
data_residency: eu         # Must stay in EU region
max_tier: premium          # Can use up to premium models
pii_handling: redact       # Redact PII before sending to provider
```

To add a new repo:
1. Create `configs/repos/<repo-name>.yml`
2. Add an API key mapping in `configs/keys.yml`
3. Restart the broker

### API Keys (`configs/keys.yml`)

Maps API keys to repos and teams:

```yaml
keys:
  sk-fintech-abc123:
    repo: fintech-app
    team: fintech-squad
  sk-internal-xyz789:
    repo: internal-tooling
    team: platform-team
```

### Router Settings (`configs/router.yml`)

Controls the intelligent routing layer:

```yaml
router:
  strategy: mf              # RouteLLM strategy (mf, sw_ranking, bert)
  cost_threshold: 0.5       # 0.0 = always weak, 1.0 = always strong
  strong_model: "claude-sonnet"
  weak_model: "gpt-4o-mini"
```

---

## API Usage

The broker exposes two endpoints:
- `/v1/chat/completions` -- OpenAI-compatible format
- `/v1/messages` -- Anthropic-compatible format

### OpenAI Format

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-internal-xyz789" \
  -d '{
    "messages": [
      {"role": "user", "content": "Explain Python decorators briefly"}
    ]
  }'
```

### Size Hint Override

Force routing to a specific tier via the `X-Size` header:

```bash
# Force premium tier
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer sk-internal-xyz789" \
  -H "X-Size: large" \
  -d '{"messages": [{"role": "user", "content": "Refactor this complex codebase..."}]}'

# Force free/standard tier
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer sk-internal-xyz789" \
  -H "X-Size: small" \
  -d '{"messages": [{"role": "user", "content": "What is 2+2?"}]}'
```

### Streaming

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer sk-internal-xyz789" \
  -d '{"messages": [{"role": "user", "content": "Write a haiku"}], "stream": true}'
```

### Anthropic Format

```bash
curl -X POST http://localhost:8000/v1/messages \
  -H "Authorization: Bearer sk-internal-xyz789" \
  -d '{
    "model": "claude-sonnet",
    "max_tokens": 1024,
    "messages": [{"role": "user", "content": "Hello from Claude Code"}]
  }'
```

### Using with Claude Code

```bash
export ANTHROPIC_BASE_URL=http://localhost:8000/v1
export ANTHROPIC_API_KEY=sk-internal-xyz789
```

The broker accepts Anthropic-format requests at `/v1/messages`, converts them internally, routes through the compliance pipeline, and returns Anthropic-format responses.

---

## Azure Deployment

Deploys to Azure Container Apps (consumption plan -- scales to zero, pay per request).

### One-Command Deploy

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
export AZURE_API_KEY="..."
export OPENAI_API_KEY="sk-..."

./infra/deploy.sh

# Or customise
RESOURCE_GROUP=my-rg LOCATION=eastus ./infra/deploy.sh
```

The script is idempotent -- re-running updates the existing deployment.

### What Gets Created

| Resource | Purpose |
|----------|---------|
| Resource Group | Container for all resources |
| Container Registry (Basic) | Stores the Docker image |
| Log Analytics Workspace | Container App logs |
| Container App Environment | Managed environment (consumption plan) |
| Container App | The broker (0.5 CPU, 1Gi RAM, 0-5 replicas) |

### CI/CD with GitHub Actions

Automatically deploys on push to `main` via `.github/workflows/deploy.yml`.

**Required Secrets:**

| Secret | Description |
|--------|-------------|
| `AZURE_CREDENTIALS` | Service principal JSON |
| `ANTHROPIC_API_KEY` | Anthropic provider key |
| `AZURE_API_KEY` | Azure provider key |
| `OPENAI_API_KEY` | OpenAI provider key |

**Optional Variables:**

| Variable | Default | Description |
|----------|---------|-------------|
| `RESOURCE_GROUP` | `llm-broker-rg` | Azure resource group |
| `LOCATION` | `westeurope` | Azure region |
| `ENVIRONMENT_NAME` | `llm-broker` | Base name for resources |

### Manual Bicep Deployment

```bash
az group create --name llm-broker-rg --location westeurope

az deployment group create \
  --resource-group llm-broker-rg \
  --template-file infra/containerapp.bicep \
  --parameters \
    environmentName=llm-broker \
    imageName=llm-broker:latest \
    anthropicApiKey="$ANTHROPIC_API_KEY" \
    azureApiKey="$AZURE_API_KEY" \
    openaiApiKey="$OPENAI_API_KEY"
```

---

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `CONFIG_DIR` | No | `configs` | Path to YAML configuration directory |
| `ANTHROPIC_API_KEY` | For Anthropic | -- | Anthropic SaaS API key |
| `AZURE_API_KEY` | For Azure | -- | Azure AI Foundry API key |
| `OPENAI_API_KEY` | For OpenAI | -- | OpenAI SaaS API key |

## Project Structure

```
llm-broker/
|-- configs/
|   |-- providers.yml            # Provider registry + model catalog
|   |-- router.yml               # Router strategy settings
|   |-- keys.yml                 # API key -> repo/team mapping
|   +-- repos/                   # Per-repo compliance rules
|-- src/llm_broker/
|   |-- main.py                  # FastAPI app, endpoints
|   |-- config.py                # YAML loader, config singleton
|   |-- models.py                # Pydantic models
|   |-- compliance.py            # Layer 1: filtering + PII
|   |-- pii.py                   # PII detection/redaction
|   |-- router.py                # Layer 2: RouteLLM + tier selection
|   +-- proxy.py                 # Layer 3: LiteLLM dispatch + retry
|-- infra/
|   |-- containerapp.bicep       # Azure Bicep IaC
|   +-- deploy.sh                # One-command deploy script
|-- tests/
+-- .github/workflows/deploy.yml # CI/CD pipeline
```

## License

Internal tooling -- not open source.
