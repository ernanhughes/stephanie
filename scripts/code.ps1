$root = Get-Location
$targetDir = Join-Path $root "stephanie"

if (Test-Path $targetDir) {
    Get-ChildItem -Path $targetDir -Recurse -File -Include *.py | ForEach-Object {
        $relPath = $_.FullName.Substring($root.Path.Length + 1)
        "`n`"$relPath`"`n---START-OF-FILE---`n" + (Get-Content $_.FullName -Raw) + "---END-OF-FILE---`n"
    } | Out-String  | Out-File "stephanie_code_dump.txt"
} else {
    Write-Error "Directory 'stephanie' not found in current location: $root"
}