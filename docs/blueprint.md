# Blueprint: LLM Broker

> Internal Tooling PoC — Compliance-aware routing across LLM providers

## Goal

A single OpenAI-compatible endpoint that transparently routes requests to the right LLM provider based on per-repo compliance rules and cost optimization. Invisible to developers — they just code.

### Primary use case: Claude Code
Devs configure their Claude Code to point at the broker:
```bash
ANTHROPIC_BASE_URL=http://broker:8000/v1
ANTHROPIC_API_KEY=sk-fintech-abc123   # team key — maps to repo config
```
The broker identifies which repo/team by their API key — no custom headers, no hacks. Devs set two env vars and forget about compliance forever.

### Caller identification: API key → repo mapping
```yaml
# configs/keys.yml (PoC)
keys:
  sk-fintech-abc123:
    repo: fintech-app
    team: fintech-squad
  sk-internal-xyz789:
    repo: internal-tooling
    team: platform-team
```
- PoC: static YAML file with key → repo mapping
- Production: database-backed with rotation, revocation, audit trail, eventually Azure AD/SSO

### Future use cases: Custom apps & tools
Any OpenAI-compatible client (custom apps, scripts, CI pipelines, internal tools) can use the same endpoint. The broker becomes the single gateway for all LLM usage across the org.

## Non-Goals

- Production-grade HA / multi-region deployment
- Usage billing or chargeback dashboard
- Fine-grained RBAC / per-user auth (PoC uses API keys or no auth)
- Custom model fine-tuning or training
- GUI admin panel (config is YAML files)
- Prompt caching or semantic caching

## Acceptance Criteria

- [ ] `POST /v1/chat/completions` (OpenAI format) works end-to-end
- [ ] Request with `sk-fintech-*` API key routes only to EU providers (azure-foundry)
- [ ] Request with `sk-internal-*` API key can route to any allowed provider
- [ ] Invalid/unknown API key returns 401
- [ ] PII (email, phone, SSN) is redacted before dispatch when `pii_handling: redact`
- [ ] PII passes through when `pii_handling: allow`
- [ ] RouteLLM classifies requests as strong/weak tier locally (~1ms, no LLM call)
- [ ] Size hint (`X-Size: small|medium|large`) can override RouteLLM decision (large → always strong)
- [ ] Fallback to next-best model on provider failure (auto-retry)
- [ ] Streaming responses work end-to-end (`stream: true`)
- [ ] Tool/function calling passes through correctly
- [ ] Per-request cost and latency logged (stdout/structured JSON)
- [ ] Unknown repo returns 400 with clear error
- [ ] `docker compose up` starts the full stack locally
- [ ] Deployed to Azure Container App with a public URL
- [ ] Orcha can call the broker endpoint as a demo

## Architecture

```
Developer Request
  │  POST /v1/chat/completions  (OpenAI format)
  │  Authorization: Bearer sk-fintech-abc123
  │  (optional) X-Size: medium
  ▼
┌─────────────────────────────────────┐
│  Layer 1: Compliance Gateway        │
│  ─ Load repo config (YAML)          │
│  ─ Filter providers by allowlist    │
│  ─ Filter by data residency         │
│  ─ Enforce tier ceiling             │
│  ─ PII detection & redaction        │
│  Output: sanitized request +        │
│          list of eligible models     │
└──────────────┬──────────────────────┘
               ▼
┌─────────────────────────────────────┐
│  Layer 2: Intelligent Router        │
│  ─ RouteLLM: strong or weak? (~1ms)│
│  ─ Size hint override from header   │
│  ─ Group eligible models by tier    │
│  ─ Pick cheapest in selected tier   │
│  ─ Build fallback chain             │
│  Output: ordered model list         │
└──────────────┬──────────────────────┘
               ▼
┌─────────────────────────────────────┐
│  Layer 3: Provider Proxy (LiteLLM)  │
│  ─ Format translation to provider  │
│    (handled by LiteLLM)            │
│  ─ Stream / tool-call passthrough   │
│  ─ Auto-retry with fallback         │
│  ─ Log cost + latency               │
└──────────────┬──────────────────────┘
               ▼
         Provider (Anthropic / Azure / OpenAI / Local)
```

### Data Flow (per request)

1. FastAPI receives OpenAI-format request with API key in `Authorization` header
2. **Key Resolver** maps API key → repo config (from `configs/keys.yml`)
3. **Compliance Gateway** loads repo config, filters providers/models, optionally redacts PII
4. **Intelligent Router** groups eligible models into strong/weak tiers → RouteLLM classifies locally (~1ms) → picks cheapest in selected tier → builds fallback chain
5. **Provider Proxy** calls LiteLLM with the top model (LiteLLM handles provider format translation), falls back on failure
6. OpenAI-format response streamed back to caller

### Provider Registry

```yaml
# configs/providers.yml
providers:
  anthropic-saas:
    litellm_prefix: "anthropic/"
    region: us
    deployment: saas
    models:
      - id: claude-haiku
        litellm_model: "anthropic/claude-3-5-haiku-20241022"
        tier: standard
        quality: 0.6
        cost_per_1k_tokens: 0.001
      - id: claude-sonnet
        litellm_model: "anthropic/claude-sonnet-4-20250514"
        tier: premium
        quality: 0.9
        cost_per_1k_tokens: 0.015

  azure-foundry:
    litellm_prefix: "azure/"
    region: eu
    deployment: managed
    models:
      - id: azure-claude
        litellm_model: "azure/claude-sonnet"
        tier: premium
        quality: 0.9
        cost_per_1k_tokens: 0.015
      - id: azure-gpt4o
        litellm_model: "azure/gpt-4o"
        tier: premium
        quality: 0.85
        cost_per_1k_tokens: 0.01

  openai-saas:
    litellm_prefix: "openai/"
    region: us
    deployment: saas
    models:
      - id: gpt-4o
        litellm_model: "openai/gpt-4o"
        tier: premium
        quality: 0.85
        cost_per_1k_tokens: 0.01
      - id: gpt-4o-mini
        litellm_model: "openai/gpt-4o-mini"
        tier: standard
        quality: 0.5
        cost_per_1k_tokens: 0.0006

  local:
    litellm_prefix: "ollama/"
    region: local
    deployment: on-prem
    models:
      - id: qwen3
        litellm_model: "ollama/qwen3"
        tier: free
        quality: 0.4
        cost_per_1k_tokens: 0.0
      - id: llama
        litellm_model: "ollama/llama3.1"
        tier: free
        quality: 0.45
        cost_per_1k_tokens: 0.0
```

### Repo Config Schema

```yaml
# configs/repos/fintech-app.yml
repo: fintech-app
allowed_providers:
  - azure-foundry
data_residency: eu        # only providers in this region
max_tier: premium          # ceiling (free < standard < premium)
pii_handling: redact       # redact | allow
```

### PII Patterns

Regex-based, pluggable via config:

| Type | Pattern | Replacement |
|------|---------|-------------|
| Email | `\b[\w.+-]+@[\w-]+\.[\w.-]+\b` | `[REDACTED_EMAIL]` |
| Phone | `\+?[\d\s\-().]{7,15}` | `[REDACTED_PHONE]` |
| SSN | `\b\d{3}-\d{2}-\d{4}\b` | `[REDACTED_SSN]` |

### RouteLLM Integration (Layer 2)

**Library:** [lm-sys/RouteLLM](https://github.com/lm-sys/RouteLLM) (MIT license, free, runs locally)

**How it works:**
1. Eligible models (from Layer 1) are grouped into two tiers:
   - **Strong:** premium tier models (Sonnet, GPT-4o, Azure Claude)
   - **Weak:** standard/free tier models (Haiku, GPT-4o-mini, Qwen3, Llama)
2. RouteLLM's pre-trained classifier analyzes the prompt locally (~1ms) and returns a routing decision
3. Within the selected tier, pick the cheapest available model
4. Build fallback chain: remaining models in same tier → models in other tier

**Router strategies available** (configurable in `router.yml`):
- `mf` — Matrix factorization (recommended, good balance)
- `sw_ranking` — Similarity-weighted ranking
- `bert` — BERT classifier (most accurate, slightly heavier)
- `causal_llm` — Causal LLM router (heaviest)

**Size hint override:**
- `X-Size: large` → always strong tier (skip RouteLLM)
- `X-Size: small` → always weak tier (skip RouteLLM)
- `X-Size: medium` or absent → let RouteLLM decide

**Cost threshold:** Configurable `cost_threshold` (0.0–1.0) in `router.yml` controls how aggressively to route to weak models. Lower = more cost savings, higher = more quality.

```yaml
# configs/router.yml
router:
  strategy: mf                    # RouteLLM strategy
  cost_threshold: 0.5             # 0.0 = always weak, 1.0 = always strong
  strong_model: "claude-sonnet"   # reference for RouteLLM calibration
  weak_model: "gpt-4o-mini"      # reference for RouteLLM calibration
```

## File Layout

```
llm-broker/
├── docker-compose.yml
├── Dockerfile
├── pyproject.toml
├── README.md
├── configs/
│   ├── providers.yml          # Provider registry + model catalog
│   ├── router.yml             # Global settings (weights, classifier model)
│   ├── keys.yml               # API key → repo/team mapping
│   └── repos/
│       ├── fintech-app.yml
│       └── internal-tooling.yml
├── src/
│   └── llm_broker/
│       ├── __init__.py
│       ├── main.py            # FastAPI app, /v1/chat/completions
│       ├── config.py          # YAML loader, pydantic models
│       ├── compliance.py      # Layer 1: filtering + PII
│       ├── router.py          # Layer 2: RouteLLM + tier selection
│       ├── proxy.py           # Layer 3: LiteLLM dispatch + retry
│       ├── models.py          # Shared pydantic types
│       └── pii.py             # PII detection/redaction
└── tests/
    ├── conftest.py
    ├── test_compliance.py
    ├── test_router.py
    ├── test_proxy.py
    ├── test_pii.py
    └── test_e2e.py
```

## Milestones

### Milestone 1: Project Skeleton + Config Loading

**Intent:** Bootable FastAPI app with OpenAI-compatible endpoint, config loading, key resolution.

**Files:**
- `pyproject.toml`, `Dockerfile`, `docker-compose.yml`
- `configs/providers.yml`, `configs/repos/*.yml`, `configs/router.yml`, `configs/keys.yml`
- `src/llm_broker/main.py`, `config.py`, `models.py`
- `tests/conftest.py`, `tests/test_config.py`

**Verification:**
```bash
docker compose up --build    # app starts
curl http://localhost:8000/health  # returns {"status": "ok", "repos": [...]}
# Endpoint accepts requests (stub 501 until layers wired):
curl -X POST http://localhost:8000/v1/chat/completions -d '{}'
pytest tests/ -v  # config loading + key resolution works
```

### Milestone 2: Compliance Gateway (Layer 1)

**Intent:** Filter models by repo config. PII detection and redaction.

**Files:**
- `src/llm_broker/compliance.py`
- `src/llm_broker/pii.py`
- `tests/test_compliance.py`, `tests/test_pii.py`

**Verification:**
```bash
pytest tests/test_compliance.py -v   # provider filtering works
pytest tests/test_pii.py -v          # email/phone/SSN redacted
# Manual: POST with PII in message body, confirm redacted in logs
```

### Milestone 3: Intelligent Router (Layer 2)

**Intent:** RouteLLM integration — classify strong/weak locally, tier grouping, model selection, size hint override.

**Dependencies:** `routellm` (pip install routellm)

**Files:**
- `src/llm_broker/router.py`
- `configs/router.yml` (strategy, cost_threshold, tier mappings)
- `tests/test_router.py`

**Verification:**
```bash
pytest tests/test_router.py -v
# Test: complex prompt → strong tier model selected
# Test: simple prompt → weak tier model selected
# Test: X-Size: large → strong tier regardless
# Test: X-Size: small → weak tier regardless
# Test: only weak models eligible → weak regardless of classification
```

### Milestone 4: Provider Proxy (Layer 3)

**Intent:** LiteLLM integration — dispatch, streaming, tool calls, auto-retry, cost logging.

**Files:**
- `src/llm_broker/proxy.py`
- `tests/test_proxy.py`

**Verification:**
```bash
pytest tests/test_proxy.py -v
# Manual: stream=true request, confirm chunks arrive
# Manual: tool_call request, confirm pass-through
```

### Milestone 5: End-to-End Integration + Docker

**Intent:** Wire all 3 layers together. Full request pipeline. Docker Compose for local dev.

**Files:**
- `src/llm_broker/main.py` (wire layers)
- `docker-compose.yml` (finalize)
- `tests/test_e2e.py`

**Verification:**
```bash
docker compose up --build
# E2E: fintech-app request → EU provider only, PII redacted
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-fintech-abc123" \
  -d '{"messages": [{"role": "user", "content": "Email john@acme.com about the fix"}]}'
# Confirm: API key resolved to fintech-app, routed to azure-foundry, email redacted
pytest tests/ -v
```

### Milestone 6: Azure Container App Deployment

**Intent:** Deploy to Azure Container App for team-wide access and demo from Orcha.

**Files:**
- `infra/containerapp.bicep` (or `az cli` script)
- `infra/deploy.sh` — one-command deploy script
- `.github/workflows/deploy.yml` (optional: CI/CD on push to main)
- `README.md` — add deployment instructions

**Infrastructure:**
- Azure Container App (consumption plan — pay per request, scales to zero)
- Azure Container Registry for the Docker image
- Environment variables for provider API keys (Container App secrets)
- Ingress: external with HTTPS (auto TLS)

**Verification:**
```bash
# Deploy
./infra/deploy.sh

# Smoke test against live URL
curl -X POST https://llm-broker.<region>.azurecontainerapps.io/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-fintech-abc123" \
  -d '{"messages": [{"role": "user", "content": "Hello from the cloud"}]}'

# Demo: point Orcha at the broker
ANTHROPIC_BASE_URL=https://llm-broker.<region>.azurecontainerapps.io/v1
```

## Risks & Unknowns

| Risk | Impact | Mitigation |
|------|--------|------------|
| LiteLLM streaming compatibility varies by provider | Broken streaming for some providers | Test each provider in M4, document gaps |
| RouteLLM pre-trained models may not perfectly match our provider set | Suboptimal routing for niche models | Use `mf` strategy + tune `cost_threshold`; can fine-tune on own data later |
| Local Ollama models may not be available in all environments | Fallback chain breaks | Graceful degradation — skip unavailable providers |
| Azure Foundry requires specific auth (managed identity vs key) | Can't connect in PoC | Start with API key auth, document managed identity upgrade path |
| PII regex has false positives (phone numbers vs IDs) | Over-redaction | Conservative patterns + allow per-repo opt-out |

## Open Questions

~~1. **Auth for the broker itself** — API key from env var~~
~~2. **Ollama endpoint** — Shared instance~~
~~3. **RouteLLM model download** — Bundle in Docker image~~

All resolved. No open questions remain.

## Production Roadmap

> What's needed beyond the PoC to run this for real.

### Phase 1: Security & Access Control

| Item | PoC | Production |
|------|-----|------------|
| Authentication | Static API keys in `keys.yml` | Database-backed keys with rotation, revocation, expiry |
| Authorization | Key → repo mapping (implicit) | RBAC: which teams can use which tier/provider |
| Key management | YAML file in repo | Azure Key Vault / HashiCorp Vault |
| Identity | API key → repo lookup | Azure AD / SSO integration, API key as bridge |
| Onboarding | Manually add key to YAML | Self-service: team requests key via CLI or portal |

### Phase 2: Observability & Monitoring

| Item | PoC | Production |
|------|-----|------------|
| Logging | stdout JSON | Structured logging → ELK / Datadog / Azure Monitor |
| Tracing | None | OpenTelemetry: trace each request through all 3 layers + provider call |
| Metrics | None | Prometheus: request count, latency p50/p95/p99, error rate, model usage |
| Dashboards | None | Grafana / Datadog dashboards: cost per team, routing decisions, provider health |
| Alerting | None | PagerDuty / Teams alerts: provider down, error spike, budget threshold |

### Phase 3: Cost Controls & Budgets

| Item | PoC | Production |
|------|-----|------------|
| Cost tracking | Log per-request cost | Aggregate cost per team/repo/day/month in a database |
| Budget limits | None | Configurable ceiling per team/repo — reject or downgrade when exceeded |
| Alerts | None | Notify team leads at 80%/90%/100% budget thresholds |
| Spend dashboard | None | Self-service UI: teams see their own usage and cost trends |
| Chargeback | None | Monthly cost reports per cost center for internal billing |

### Phase 4: Rate Limiting & Traffic Management

| Item | PoC | Production |
|------|-----|------------|
| Rate limiting | None | Token bucket per team/repo (requests/min + tokens/min) |
| Queuing | None | Queue requests when rate-limited instead of rejecting (with timeout) |
| Priority lanes | None | Critical repos/teams get priority; batch jobs get lower priority |
| Circuit breaker | LiteLLM retry | Per-provider circuit breaker: stop sending traffic to unhealthy providers |

### Phase 5: Caching

| Item | PoC | Production |
|------|-----|------------|
| Prompt caching | None | Exact-match cache (hash of messages → cached response) |
| Semantic caching | None | Embedding-based similarity cache (similar prompts → cached response) |
| Cache invalidation | N/A | TTL-based + manual flush per repo |
| Expected savings | N/A | 20-40% cost reduction for repetitive workloads (CI, code review) |

### Phase 6: PII Hardening

| Item | PoC | Production |
|------|-----|------------|
| Detection | Regex patterns (email, phone, SSN) | Microsoft Presidio or dedicated NER model (fewer false positives) |
| Patterns | 3 built-in | Configurable per repo: credit cards, IBANs, national IDs, custom patterns |
| Audit trail | None | Log what was redacted (type + position, not the value) for compliance proof |
| Response scanning | None | Scan provider responses for PII leakage before returning to caller |
| Testing | Unit tests | Red-team PII bypass testing, edge case corpus |

### Phase 7: Compliance Audit & Governance

| Item | PoC | Production |
|------|-----|------------|
| Routing proof | Implicit in code logic | Immutable audit log: every request → which provider, why, compliance check result |
| Export | None | Exportable audit reports for regulators / internal compliance |
| Config changes | Git commits | Change audit trail: who changed what config, when, with approval |
| Data residency proof | Config-driven filtering | Audit log proves no EU data left EU region |
| Policy-as-code | YAML files | OPA/Rego policies for complex compliance rules |

### Phase 8: High Availability & Deployment

| Item | PoC | Production |
|------|-----|------------|
| Deployment | Single Docker container | Kubernetes / Azure Container Apps |
| Scaling | Single instance | Horizontal auto-scaling based on request volume |
| Health checks | `/health` endpoint | Deep health probes (check each provider connectivity) |
| Zero-downtime deploys | N/A | Rolling deploys, blue-green, or canary |
| Multi-region | N/A | EU instance for EU data, US instance for US data |
| DR | N/A | Multi-AZ deployment, failover runbook |

### Phase 9: Advanced Routing

| Item | PoC | Production |
|------|-----|------------|
| A/B testing | None | Route % of traffic to new models, compare quality/cost |
| Quality feedback | None | Devs rate responses → feed back into RouteLLM threshold tuning |
| Custom RouteLLM training | Pre-trained models | Fine-tune on internal usage data for better routing accuracy |
| Prompt optimization | None | Auto-compress/rewrite prompts to reduce token usage |
| Model benchmarking | Static quality scores | Continuous eval: run test prompts against models, update scores automatically |

### Phase 10: Developer Experience

| Item | PoC | Production |
|------|-----|------------|
| Onboarding | Manual env var setup | `llm-broker configure` CLI that sets up Claude Code env vars |
| SDK | Raw HTTP | Thin Python/TS SDK wrapping the broker API |
| Status page | None | `/status` showing provider health, current routing, your team's usage |
| Self-service config | Edit YAML + PR | Web UI or CLI for teams to manage their own repo config |
| Documentation | README | Developer portal with guides, examples, troubleshooting |

### The honest summary

The PoC covers the **happy path** — request comes in, gets routed correctly, response comes back. Production is mostly about what happens when things go wrong (provider down, budget exceeded, PII leak) and proving to compliance that the system works as advertised. The core routing logic won't change much. Most production work is infrastructure and operational tooling around it.

**Suggested production order:** Phases 1-3 first (security, observability, cost controls) — these are table stakes. Phases 4-7 next (rate limiting, caching, PII hardening, audit) — needed before handling real customer data. Phases 8-10 last (HA, advanced routing, DX) — scale and polish.

---

Next: `/flow "Milestone 1 — Project skeleton, config loading, Docker Compose, health endpoint"`
