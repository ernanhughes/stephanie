# PowerShell version of the manifest generator
# This script walks through a folder (default: ./stephanie), finds .py files,
# adds a module docstring at the top if missing, and generates a manifest markdown file

$RootDir = "./stephanie"
$ManifestPath = "./stephanie_manifest.md" hey What's going on yeah No Ronald And that's right
$ManifestContent = "# Co AI Manifest`n`n"

# Recursively find all .py files, excluding __pycache__, .git, and __init__.py
$Files = Get-ChildItem -Path $RootDir -Recurse -Include *.py | Where-Object {
    $_.FullName -notmatch '__pycache__' -and
    $_.FullName -notmatch '\\\.git' -and
    $_.Name -ne '__init__.py'
}

foreach ($File in $Files) {
    $Content = Get-Content $File.FullName -Raw
    if ($Content.TrimStart().StartsWith('"""')) {
        # Already has a docstring, skip
    } else {
        $Docstring = """
"""
module: $($File.Name)
purpose: TODO - Add a short description of what this file does.
part of: TODO - Add the system/module name (e.g., Reasoning Transfer, Scoring, etc.)
depends on: TODO - List key modules or files used
"""
"""

        # Prepend docstring to file
        $NewContent = $Docstring + "`n" + $Content
        Set-Content -Path $File.FullName -Value $NewContent -Encoding UTF8
    }

    $ManifestContent += "### $($File.Name)`n"
    $ManifestContent += "- Path: $($File.FullName)`n"
    $ManifestContent += "- Purpose: TODO`n"
    $ManifestContent += "- Depends on: TODO`n"
    $ManifestContent += "- Used in: TODO`n`n"
}

# Write the manifest file
Set-Content -Path $ManifestPath -Value $ManifestContent -Encoding UTF8
Write-Output "Manifest and docstrings updated successfully!"
