# SC3 User Guide

Run from `C:\Projects\stephanie` in PowerShell. Bun and Python 3.13 must be available; the checked Worker source is expected at `C:\Projects\worker`. Set the local module path for CLI use:

```powershell
$env:PYTHONPATH = 'C:\Projects\stephanie;C:\Projects\praxis;C:\Projects\ai'
python -m stephanie.improvement.cli --help
```

Create a new isolated experiment root for a fresh run; do not reuse a root containing frozen artifacts:

```powershell
$root = 'data\stephanie_convergence03_new'
python -m stephanie.improvement.cli --root $root sc3-hypothesis
python -m stephanie.improvement.cli --root $root sc3-freeze
python -m stephanie.improvement.cli --root $root sc3-run
python -m stephanie.improvement.cli --root $root sc3-report
python -m stephanie.improvement.cli --root $root sc3-retrieve 'json structured output prior unsuccessful candidate'
```

The hypothesis command is once-only and makes one local shared-AI call; it stores provider evidence privately and prints only metadata. Do not rerun it after an artifact is created. The separately bounded genuine Worker probe uses only the already installed local-only Worker model config. It performs a readiness request and stops on readiness failure; otherwise it performs one baseline/candidate pair. The verified route has a ceiling of four Worker calls (one readiness plus up to two calls per strategy), while observed execution used three. Across the SC3 AI hypothesis and Worker live lanes, four model invocations were observed, below the six-call ceiling. No model weights are downloaded.

```powershell
$env:WORKER_MODELS_CONFIG = 'C:\Projects\worker\worker.models.yaml'
python -m stephanie.improvement.cli --root 'data\stephanie_convergence03_live_new' sc3-live-probe
```

`sc3-recompare` applies Praxis again to preserved Worker records and does not execute Worker. `sc3-report` emits summaries and artifact pointers; full model responses remain in local result artifacts. Review `frozen-protocol.json`, `frozen-evaluator.json`, `worker-results.json`, `praxis-c*.json`, `integrated-results.json`, and the retrieval artifacts under the chosen root. Results do not authorize Mission acceptance or promotion. Worker source remains the separately checked out `C:\Projects\worker`; its required packages must be installed for the imported source tree.
