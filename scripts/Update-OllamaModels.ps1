# Update-OllamaModels.ps1

Write-Host "🔍 Fetching list of local Ollama models..."

# Get the list of installed models
$modelsText = ollama list
$lines = $modelsText -split "`n"

# Skip header line and parse model names
$models = @()
foreach ($line in $lines[1..($lines.Count - 1)]) {
    $name = ($line -split "\s+")[0]
    if ($name -and ($models -notcontains $name)) {
        $models += $name
    }
}

if ($models.Count -eq 0) {
    Write-Host "⚠️  No models found. Are you sure Ollama is installed and models are downloaded?"
    exit
}

Write-Host "📦 Found $($models.Count) models. Updating each one..."

foreach ($model in $models) {
    Write-Host "`n⬇️  Pulling latest for model: $model ..."
    ollama pull $model
}

Write-Host "`n✅ All local Ollama models are now up to date!"
