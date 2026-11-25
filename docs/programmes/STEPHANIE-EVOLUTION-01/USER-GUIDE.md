# User guide

Run commands from `C:/Projects/stephanie` using the 3.13 environment. The default artifact and isolated SQLite location is `data/stephanie_evolution01_final`; override it with `--root <path>`.

```powershell
venv\Scripts\python.exe -m stephanie.improvement import-harness
venv\Scripts\python.exe -m stephanie.improvement discover
venv\Scripts\python.exe -m stephanie.improvement ai-investigate --provider ollama --model ornith-1.5:9b
venv\Scripts\python.exe -m stephanie.improvement construct
venv\Scripts\python.exe -m stephanie.improvement validate d80cd1cbb12e2f2e75107745
venv\Scripts\python.exe -m stephanie.improvement evaluate d80cd1cbb12e2f2e75107745
venv\Scripts\python.exe -m stephanie.improvement inspect d80cd1cbb12e2f2e75107745
venv\Scripts\python.exe -m stephanie.improvement recommend d80cd1cbb12e2f2e75107745
venv\Scripts\python.exe -m stephanie.improvement retrieve "Harness evidence acceptance evaluation"
venv\Scripts\python.exe -m stephanie.improvement ai-evidence
```

`ai-investigate` makes a real local model call; it is capped at one call/120 seconds and is not retried automatically. Other commands use the imported evidence and local deterministic evaluator. Recommendations are fixture-scoped and never promote operational changes.
