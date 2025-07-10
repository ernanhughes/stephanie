# clean-project.ps1

$folders = @("models")

foreach ($folder in $folders) {
    if (Test-Path $folder) {
        $items = Get-ChildItem -Path $folder -Recurse -Force
        $count = $items.Count

        if ($count -gt 0) {
            Write-Host "🧹 Deleting $count items from '$folder'..."
            $items | Remove-Item -Force -Recurse -ErrorAction SilentlyContinue
            Write-Host "✅ Deleted $count items from '$folder'.`n"
        } else {
            Write-Host "ℹ️ '$folder' is already empty.`n"
        }
    } else {
        Write-Host "⚠️ '$folder' does not exist.`n"
    }
}

Write-Host "✅ Project cleanup complete."
