# Writer Runtime Integration

Stephanie can discover and (optionally) execute **Writer** runtimes from a local
Writer checkout.

## Configuration

Enable the optional service in your Hydra service config:

```yaml
services:
  writer_runtime:
    enabled: true
    cls: stephanie.integrations.writer_runtime.WriterRuntimeService
    dependencies: []
    args:
      cfg:
        project_root: "C:/Projects/Writer"
        writer_executable: null   # or explicit path to writer.exe
        timeout_seconds: 30
        enable_execution: false    # discovery only by default
```

Discovery works through the Writer CLI (`writer info ... --json`) and does not
require importing Writer. Execution is opt-in and requires the Writer packages to
be importable from the configured `project_root`. When execution is enabled the
bridge adds each `C:/Projects/Writer/packages/*` directory to `sys.path` and
builds the Writer `Runtime` gateway in-process, so the runtime facades (e.g.
`writer_ai`) must be installed or importable and the Writer database schema
must be provisioned. If Writer is missing, execution returns an explicit
`unavailable`/`disabled` result — Stephanie never silently falls back to
another engine.

## Programmatic use

```python
from stephanie.integrations.writer_runtime import WriterRuntimeConfig, WriterRuntimeBridge

bridge = WriterRuntimeBridge(WriterRuntimeConfig(
    project_root="C:/Projects/Writer",
    enable_execution=False,
))

print(bridge.status())
print(bridge.discovery.search("repair a sentence"))
print(bridge.discovery.describe("editorial"))
```

## SIS UI

When `writer.enabled` is true in `config/sis.yaml`, SIS exposes a **Writer** tab:

* `GET /writer` — search runtimes by task description
* `GET /writer/runtime/{name}` — describe a runtime and list its operations

Discovery is read-only. The SIS routes never execute a runtime, even when
`enable_execution` is true elsewhere.

## Supervisor

`Supervisor` exposes the same discovery surface:

```python
ok, results = supervisor.discover_writer_runtimes("repair a sentence")
```

It reads `writer` (or `writer_runtime`) from the composed Hydra config and is a
no-op ("Writer integration is not enabled") unless explicitly enabled.

## Safety notes

* Discovery is read-only and shells out with argument arrays (no shell mode).
* Execution is disabled by default; enable only after verifying the Writer
  checkout and its database configuration.
* Stephanie does not auto-approve Writer policy decisions; governed invocation
  stays with Writer's own authority/policy layer.
