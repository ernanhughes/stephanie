# DB connection (adjust as needed)
$db_host = 'localhost'
$port = '5432'
$user = 'co'
$db   = 'co'

$env:PGPASSWORD = 'co'
# Optional: avoid password prompt
# $env:PGPASSWORD = 'your_password_here'

$pgbin = 'C:\Program Files\PostgreSQL\17\bin'
# Where to save
$out = 'schema_raw.sql'

# Dump schema and pretty-print
& "$pgbin\pg_dump.exe" -h $db_host -p $port -U $user -d $db `
  -s -C -c --if-exists --no-owner --no-privileges `
| sqlformat - --reindent --keywords upper `
| Set-Content -Encoding utf8 .\$out

# Remove top-level SET lines and squeeze blank lines
$content = Get-Content schema_raw.sql -Raw
$content = [regex]::Replace($content, '^(SET .*?;\r?\n)+', '', 'Multiline')
$content = [regex]::Replace($content, '(\r?\n){3,}', "`r`n`r`n")
Set-Content -Path schema_raw.sql -Value $content -Encoding UTF8
