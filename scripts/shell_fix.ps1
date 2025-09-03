
Add-Content -Path $PROFILE -Value '[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()'
setx PYTHONUTF8 1
setx PYTHONIOENCODING utf-8
