# User guide

Set `PYTHONPATH` to Stephanie and shared AI source as needed; add `C:\Projects\praxis` to PYTHONPATH for canonical Praxis checks.

```powershell
$env:PYTHONPATH='C:\Projects\ai\src;C:\Projects\stephanie;C:\Projects\praxis'
python -m stephanie.improvement integration-check
python -m stephanie.improvement worker-strategies
python -m stephanie.improvement worker-executions --capture
python -m stephanie.improvement mission-evidence --limit 20 --capture
python -m stephanie.improvement investigate-worker-issue
python -m stephanie.improvement retrieve "worker empty output"
python -m stephanie.improvement propose-worker-diagnostic
python -m stephanie.improvement inspect-investigation worker-empty-output-se2q-reference-v1
python -m stephanie.improvement ai-evidence
```

`--capture` writes bounded source references under the selected `--root`; it does not alter source systems. Raw AI bodies require deliberate `ai-evidence --include-raw`. Worker/Mission endpoints are GET-only. Praxis canonical callable is available to Python callers via `compare_with_praxis`; this CLI does not manufacture cohorts.
