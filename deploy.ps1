$ErrorActionPreference = "Stop"

$ResourceGroup = "rag-rg"
$Location = "eastus"
$AcrName = "ragacrmilug0430"
$KvName = "ragkvmilug0430"
$ApiImage = "$AcrName.azurecr.io/rag-api:latest"
$UiImage = "$AcrName.azurecr.io/rag-ui:latest"

function Get-ProjectEnvValue {
    param([Parameter(Mandatory = $true)][string]$Name)

    $value = [Environment]::GetEnvironmentVariable($Name)
    if ($value) {
        return $value
    }

    $envPath = Join-Path $PSScriptRoot ".env"
    if (-not (Test-Path $envPath)) {
        return ""
    }

    foreach ($line in Get-Content $envPath) {
        if ($line -match "^\s*#" -or $line -notmatch "=") {
            continue
        }

        $parts = $line.Split("=", 2)
        if ($parts[0].Trim() -eq $Name) {
            return $parts[1].Trim().Trim('"').Trim("'")
        }
    }

    return ""
}

$OpenAiApiKey = Get-ProjectEnvValue "OPENAI_API_KEY"
$AnthropicApiKey = Get-ProjectEnvValue "ANTHROPIC_API_KEY"

if (-not $OpenAiApiKey) {
    throw "OPENAI_API_KEY must be set in the environment or .env before deployment."
}

if (-not $AnthropicApiKey) {
    throw "ANTHROPIC_API_KEY must be set in the environment or .env before deployment."
}

az group create -n $ResourceGroup -l $Location
az deployment group create -g $ResourceGroup -f infra/main.bicep --parameters acrName=$AcrName keyVaultName=$KvName apiImage=$ApiImage uiImage=$UiImage openAiApiKey="$OpenAiApiKey" anthropicApiKey="$AnthropicApiKey"
az acr build -r $AcrName -t rag-api:latest -f api/Dockerfile .
az acr build -r $AcrName -t rag-ui:latest -f Dockerfile .
az containerapp update -n rag-api -g $ResourceGroup --image $ApiImage
az containerapp update -n rag-ui -g $ResourceGroup --image $UiImage
$ApiUrl = az containerapp show -n rag-api -g $ResourceGroup --query properties.configuration.ingress.fqdn -o tsv
$UiUrl = az containerapp show -n rag-ui -g $ResourceGroup --query properties.configuration.ingress.fqdn -o tsv
Write-Host "API URL: https://$ApiUrl"
Write-Host "UI URL: https://$UiUrl"
