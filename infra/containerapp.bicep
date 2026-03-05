// ---------------------------------------------------------------------------
// Azure Bicep template for LLM Broker — Azure Container App deployment
//
// Provisions:
//   1. Azure Container Registry (Basic SKU)
//   2. Container App Environment (consumption plan)
//   3. Container App with external HTTPS ingress on port 8000
//
// Usage:
//   az deployment group create \
//     --resource-group <rg-name> \
//     --template-file infra/containerapp.bicep \
//     --parameters environmentName=llm-broker \
//                  imageName=llm-broker:latest \
//                  anthropicApiKey=<key> \
//                  azureApiKey=<key> \
//                  openaiApiKey=<key>
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Parameters
// ---------------------------------------------------------------------------

@description('Azure region for all resources')
param location string = resourceGroup().location

@description('Base name used to derive resource names (Container App, ACR, environment)')
param environmentName string = 'llm-broker'

@description('Full image name including tag (e.g. llm-broker:latest). Registry prefix is added automatically.')
param imageName string = 'llm-broker:latest'

@description('Minimum number of replicas (0 allows scale-to-zero)')
@minValue(0)
@maxValue(5)
param minReplicas int = 0

@description('Maximum number of replicas')
@minValue(1)
@maxValue(5)
param maxReplicas int = 5

@description('Number of concurrent HTTP requests that trigger a new replica')
param httpScaleRule int = 10

@secure()
@description('Anthropic API key for the anthropic-saas provider')
param anthropicApiKey string = ''

@secure()
@description('Azure API key for the azure-foundry provider')
param azureApiKey string = ''

@secure()
@description('OpenAI API key for the openai-saas provider')
param openaiApiKey string = ''

// ---------------------------------------------------------------------------
// Variables — derived names
// ---------------------------------------------------------------------------

// ACR names must be alphanumeric, 5-50 characters, globally unique.
// Strip hyphens and append 'acr' for uniqueness.
var acrName = '${replace(environmentName, '-', '')}acr${uniqueString(resourceGroup().id)}'
var containerAppEnvName = '${environmentName}-env'
var containerAppName = environmentName
var logAnalyticsName = '${environmentName}-logs'

// ---------------------------------------------------------------------------
// Log Analytics workspace (required by Container App Environment)
// ---------------------------------------------------------------------------

resource logAnalytics 'Microsoft.OperationalInsights/workspaces@2022-10-01' = {
  name: logAnalyticsName
  location: location
  properties: {
    sku: {
      name: 'PerGB2018'
    }
    retentionInDays: 30
  }
}

// ---------------------------------------------------------------------------
// Azure Container Registry (Basic SKU — sufficient for PoC)
// ---------------------------------------------------------------------------

resource acr 'Microsoft.ContainerRegistry/registries@2023-07-01' = {
  name: acrName
  location: location
  sku: {
    name: 'Basic'
  }
  properties: {
    adminUserEnabled: true  // Required for Container App to pull images
  }
}

// ---------------------------------------------------------------------------
// Container App Environment (consumption plan — pay per request)
// ---------------------------------------------------------------------------

resource containerAppEnv 'Microsoft.App/managedEnvironments@2023-05-01' = {
  name: containerAppEnvName
  location: location
  properties: {
    appLogsConfiguration: {
      destination: 'log-analytics'
      logAnalyticsConfiguration: {
        customerId: logAnalytics.properties.customerId
        sharedKey: logAnalytics.listKeys().primarySharedKey
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Container App
// ---------------------------------------------------------------------------

resource containerApp 'Microsoft.App/containerApps@2023-05-01' = {
  name: containerAppName
  location: location
  properties: {
    managedEnvironmentId: containerAppEnv.id
    configuration: {
      // -- External ingress with auto-TLS --
      ingress: {
        external: true
        targetPort: 8000
        transport: 'auto'
        allowInsecure: false
      }

      // -- ACR credentials --
      registries: [
        {
          server: acr.properties.loginServer
          username: acr.listCredentials().username
          passwordSecretRef: 'acr-password'
        }
      ]

      // -- Secrets (provider API keys + ACR password) --
      secrets: [
        {
          name: 'acr-password'
          value: acr.listCredentials().passwords[0].value
        }
        {
          name: 'anthropic-api-key'
          value: anthropicApiKey
        }
        {
          name: 'azure-api-key'
          value: azureApiKey
        }
        {
          name: 'openai-api-key'
          value: openaiApiKey
        }
      ]
    }

    template: {
      containers: [
        {
          name: containerAppName
          image: '${acr.properties.loginServer}/${imageName}'
          resources: {
            cpu: json('0.5')
            memory: '1Gi'
          }
          env: [
            {
              // CONFIG_DIR tells the app where to find YAML configs.
              // Configs are baked into the Docker image under /app/configs.
              name: 'CONFIG_DIR'
              value: '/app/configs'
            }
            {
              name: 'ANTHROPIC_API_KEY'
              secretRef: 'anthropic-api-key'
            }
            {
              name: 'AZURE_API_KEY'
              secretRef: 'azure-api-key'
            }
            {
              name: 'OPENAI_API_KEY'
              secretRef: 'openai-api-key'
            }
          ]
        }
      ]

      // -- Scaling: 0-5 replicas based on concurrent HTTP requests --
      scale: {
        minReplicas: minReplicas
        maxReplicas: maxReplicas
        rules: [
          {
            name: 'http-scale-rule'
            http: {
              metadata: {
                concurrentRequests: string(httpScaleRule)
              }
            }
          }
        ]
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Outputs
// ---------------------------------------------------------------------------

@description('The FQDN of the deployed Container App')
output appUrl string = 'https://${containerApp.properties.configuration.ingress.fqdn}'

@description('The ACR login server')
output acrLoginServer string = acr.properties.loginServer

@description('The Container App name')
output containerAppName string = containerApp.name

@description('The resource group used')
output resourceGroup string = resourceGroup().name
