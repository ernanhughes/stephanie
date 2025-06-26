param(
    [string]$SubdirPath = ".",
    [string]$OutputFile = "merged_code.md"
)

# Resolve full path
$SubdirPath = Resolve-Path $SubdirPath
$OutputPath = Join-Path $SubdirPath $OutputFile

# Initialize output file
"<!-- Merged Python Code Files -->`n" | Out-File -FilePath $OutputPath -Encoding utf8

# Get all .py files recursively
$files = Get-ChildItem -Path $SubdirPath -Recurse -Filter *.py | Sort-Object FullName

foreach ($file in $files) {
    $relativePath = $file.FullName.Replace($SubdirPath, "").TrimStart("\","/")
    Add-Content -Path $OutputPath -Value "`n## File: $relativePath`n"
    Add-Content -Path $OutputPath -Value "```python"
    Get-Content -Path $file.FullName | Add-Content -Path $OutputPath
    Add-Content -Path $OutputPath -Value "````n"
}

Write-Host "✅ Merged $($files.Count) Python files into: $OutputPath"
