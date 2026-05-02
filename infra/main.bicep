targetScope = 'resourceGroup'

@description('Azure region for all resources.')
param location string = resourceGroup().location

@secure()
@description('OpenAI API key stored in Key Vault and exposed to the API container.')
param openAiApiKey string

@secure()
@description('Anthropic API key stored in Key Vault and exposed to the API container.')
param anthropicApiKey string

@description('Container Apps environment name.')
param containerAppEnvironmentName string = 'rag-env'

@description('API Container App name.')
param apiContainerAppName string = 'rag-api'

@description('UI Container App name.')
param uiContainerAppName string = 'rag-ui'

@description('Azure Container Registry name.')
param acrName string = 'ragacrmilug0430'

@description('Azure Key Vault name.')
param keyVaultName string = 'rag-kv-milug-0430'

@description('Azure Files share name used for Chroma persistence.')
param storageShareName string = 'chroma-data'

@description('Storage account name for Azure Files.')
param storageAccountName string = toLower('rag${uniqueString(resourceGroup().id)}')

@description('API image reference. The deploy script updates this to the freshly built ACR image.')
param apiImage string = '${acrName}.azurecr.io/rag-api:latest'

@description('Placeholder UI image.')
param uiImage string = '${acrName}.azurecr.io/rag-ui:latest'

resource logAnalytics 'Microsoft.OperationalInsights/workspaces@2023-09-01' = {
  name: 'rag-law'
  location: location
  properties: {
    sku: {
      name: 'PerGB2018'
    }
    retentionInDays: 30
  }
}

resource acr 'Microsoft.ContainerRegistry/registries@2023-07-01' = {
  name: acrName
  location: location
  sku: {
    name: 'Standard'
  }
  properties: {
    adminUserEnabled: false
  }
}

resource storageAccount 'Microsoft.Storage/storageAccounts@2023-05-01' = {
  name: storageAccountName
  location: location
  kind: 'StorageV2'
  sku: {
    name: 'Standard_LRS'
  }
  properties: {
    accessTier: 'Hot'
    minimumTlsVersion: 'TLS1_2'
    supportsHttpsTrafficOnly: true
  }
}

resource fileService 'Microsoft.Storage/storageAccounts/fileServices@2023-05-01' = {
  parent: storageAccount
  name: 'default'
}

resource chromaShare 'Microsoft.Storage/storageAccounts/fileServices/shares@2023-05-01' = {
  parent: fileService
  name: storageShareName
  properties: {
    enabledProtocols: 'SMB'
    shareQuota: 100
  }
}

resource containerAppIdentity 'Microsoft.ManagedIdentity/userAssignedIdentities@2023-01-31' = {
  name: 'rag-app-identity'
  location: location
}

resource keyVault 'Microsoft.KeyVault/vaults@2023-07-01' = {
  name: keyVaultName
  location: location
  properties: {
    tenantId: subscription().tenantId
    sku: {
      family: 'A'
      name: 'standard'
    }
    accessPolicies: [
      {
        tenantId: subscription().tenantId
        objectId: containerAppIdentity.properties.principalId
        permissions: {
          secrets: [
            'Get'
            'List'
          ]
        }
      }
    ]
    enableRbacAuthorization: false
    enabledForTemplateDeployment: true
    enabledForDeployment: true
    enabledForDiskEncryption: false
    publicNetworkAccess: 'Enabled'
    softDeleteRetentionInDays: 7
  }
}

resource openAiSecret 'Microsoft.KeyVault/vaults/secrets@2023-07-01' = {
  parent: keyVault
  name: 'OPENAI-API-KEY'
  properties: {
    value: openAiApiKey
  }
}

resource anthropicSecret 'Microsoft.KeyVault/vaults/secrets@2023-07-01' = {
  parent: keyVault
  name: 'ANTHROPIC-API-KEY'
  properties: {
    value: anthropicApiKey
  }
}

resource ragEnvironment 'Microsoft.App/managedEnvironments@2023-05-01' = {
  name: containerAppEnvironmentName
  location: location
  properties: {
    appLogsConfiguration: {
      destination: 'log-analytics'
      logAnalyticsConfiguration: {
        customerId: logAnalytics.properties.customerId
        sharedKey: listKeys(logAnalytics.id, logAnalytics.apiVersion).primarySharedKey
      }
    }
  }
}

resource chromaStorage 'Microsoft.App/managedEnvironments/storages@2023-05-01' = {
  parent: ragEnvironment
  name: storageShareName
  properties: {
    azureFile: {
      accessMode: 'ReadWrite'
      accountName: storageAccount.name
      accountKey: listKeys(storageAccount.id, storageAccount.apiVersion).keys[0].value
      shareName: storageShareName
    }
  }
}

resource ragApi 'Microsoft.App/containerApps@2023-05-01' = {
  name: apiContainerAppName
  location: location
  identity: {
    type: 'UserAssigned'
    userAssignedIdentities: {
      '${containerAppIdentity.id}': {}
    }
  }
  properties: {
    managedEnvironmentId: ragEnvironment.id
    configuration: {
      activeRevisionsMode: 'Single'
      ingress: {
        external: true
        targetPort: 8000
        transport: 'auto'
        allowInsecure: false
      }
      registries: [
        {
          server: acr.properties.loginServer
          identity: containerAppIdentity.id
        }
      ]
      secrets: [
        {
          name: 'openai-api-key'
          identity: containerAppIdentity.id
          keyVaultUrl: '${keyVault.properties.vaultUri}secrets/OPENAI-API-KEY'
        }
        {
          name: 'anthropic-api-key'
          identity: containerAppIdentity.id
          keyVaultUrl: '${keyVault.properties.vaultUri}secrets/ANTHROPIC-API-KEY'
        }
      ]
    }
    template: {
      containers: [
        {
          name: 'rag-api'
          image: apiImage
          env: [
            {
              name: 'OPENAI_API_KEY'
              secretRef: 'openai-api-key'
            }
            {
              name: 'ANTHROPIC_API_KEY'
              secretRef: 'anthropic-api-key'
            }
            {
              name: 'CHROMA_PATH'
              value: '/mnt/chroma'
            }
          ]
          resources: {
            cpu: ('0.5')
            memory: '1Gi'
          }
          volumeMounts: [
            {
              volumeName: 'chroma-data'
              mountPath: '/mnt/chroma'
            }
          ]
        }
      ]
      scale: {
        minReplicas: 1
        maxReplicas: 1
      }
      volumes: [
        {
          name: 'chroma-data'
          storageType: 'AzureFile'
          storageName: storageShareName
        }
      ]
    }
  }
  dependsOn: [
    chromaStorage
    openAiSecret
    anthropicSecret
  ]
}

resource ragUi 'Microsoft.App/containerApps@2023-05-01' = {
  name: uiContainerAppName
  location: location
  identity: {
    type: 'UserAssigned'
    userAssignedIdentities: {
      '${containerAppIdentity.id}': {}
    }
  }
  properties: {
    managedEnvironmentId: ragEnvironment.id
    configuration: {
      activeRevisionsMode: 'Single'
      ingress: {
        external: true
        targetPort: 3000
        transport: 'auto'
        allowInsecure: false
      }
      registries: [
        {
          server: acr.properties.loginServer
          identity: containerAppIdentity.id
        }
      ]
    }
    template: {
      containers: [
        {
          name: 'rag-ui'
          image: uiImage
          env: [
            {
              name: 'BACKEND_API_BASE_URL'
              value: 'https://${ragApi.properties.configuration.ingress.fqdn}'
            }
          ]
          resources: {
  cpu: any('0.25')
  memory: '0.5Gi'
          }
        }
      ]
      scale: {
        minReplicas: 1
        maxReplicas: 1
      }
    }
  }
}

output apiUrl string = 'https://${ragApi.properties.configuration.ingress.fqdn}'
output uiUrl string = 'https://${ragUi.properties.configuration.ingress.fqdn}'
output acrLoginServer string = acr.properties.loginServer
output keyVaultUri string = keyVault.properties.vaultUri
