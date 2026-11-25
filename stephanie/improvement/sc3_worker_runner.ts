/** SC3-only isolated Worker runner. Uses the real Worker service/API with a
 * deterministic scripted provider and isolated execution ledger. It never
 * loads author config or the operational service's ledgers. */
import { readFileSync, writeFileSync, mkdirSync } from "node:fs"
import { dirname, resolve } from "node:path"
import { createWorkerServer } from "../../../worker/src/server/server.js"
import { WorkerService } from "../../../worker/src/core/worker.js"
import { ExecutionLedger } from "../../../worker/src/core/execution-ledger.js"
import { ModelRuntime } from "../../../worker/src/models/runtime/engine.js"
import type { ModelProvider, ModelsConfig, ProviderRequest, ProviderResponse } from "../../../worker/src/models/runtime/types.js"
import { ModelExecutor } from "../../../worker/src/executors/model.js"
import { ExecutorRegistry } from "../../../worker/src/executors/types.js"

type Protocol = { experiment_id: string; protocol_digest: string; worker_source_commit: string; artifact_root: string;
  requests: Array<Record<string, any>>; provider_responses: Record<string, string[]> }
class ScriptedProvider implements ModelProvider {
  readonly calls: Array<{ task_id: string; ordinal: number }> = []
  private counts = new Map<string, number>()
  constructor(readonly responses: Record<string, string[]>) {}
  async execute(request: ProviderRequest): Promise<ProviderResponse> {
    const task = /\[\[SC3_TASK:([A-Za-z0-9._-]+)\]\]/.exec(request.prompt)?.[1]
    if (!task) throw new Error("fixture task marker missing")
    const ordinal = this.counts.get(task) ?? 0
    this.counts.set(task, ordinal + 1)
    this.calls.push({ task_id: task, ordinal: ordinal + 1 })
    if (task.startsWith("C6_FAIL")) throw new Error("SC3 deterministic provider failure")
    const sequence = this.responses[task]
    if (!sequence?.length) throw new Error(`no scripted response for ${task}`)
    const text = sequence[Math.min(ordinal, sequence.length - 1)]!
    return { text, usage: { input_tokens: Math.ceil(request.prompt.length / 4), output_tokens: Math.ceil(text.length / 4), total_tokens: Math.ceil((request.prompt.length + text.length) / 4) }, observed: { provider: "sc3-scripted-fixture", model: "deterministic-response-table" } }
  }
  async discover() { return [{ id: "deterministic-response-table" }] }
}

const protocolPath = process.argv[2]
const outputPath = process.argv[3]
if (!protocolPath || !outputPath) throw new Error("usage: bun sc3_worker_runner.ts <protocol.json> <worker-results.json>")
const protocol = JSON.parse(readFileSync(protocolPath, "utf8")) as Protocol
const root = resolve(protocol.artifact_root)
mkdirSync(root, { recursive: true })
const ledgerPath = resolve(root, "worker-execution-ledger.jsonl")
const provider = new ScriptedProvider(protocol.provider_responses)
const config: ModelsConfig = {
  schema_version: "worker.models.v1", default_policy: "sc3.local", health: { transient_ttl_ms: 30_000 },
  providers: { fixture: { type: "opencode", adapter: "chat", base_url: "http://127.0.0.1", failure_domain: "sc3-fixture" } },
  models: { fixture_model: { provider: "fixture", model: "deterministic-response-table", economic_tier: "free", data_policy: "local_only" } },
  policies: { "sc3.local": { strategy: "ordered_fallback", routes: ["fixture_model"], allow_direct_paid: false, attempts_per_route: 1, required_data_policy: "local_only" } },
}
const runtime = new ModelRuntime(config, Date.now, { fixture: provider })
const registry = new ExecutorRegistry()
registry.register(new ModelExecutor(runtime))
const ledger = new ExecutionLedger(ledgerPath, true)
const service = new WorkerService(registry, "model", {}, runtime, undefined, undefined, undefined, undefined, undefined, ledger)
const server = createWorkerServer(service, { host: "127.0.0.1", port: 0 })
const base = `http://${server.hostname}:${server.port}`
try {
  const health = await (await fetch(`${base}/health/transport`)).json()
  const strategies = await (await fetch(`${base}/v1/strategies`)).json()
  const results: Array<{ request: Record<string, any>; response: Record<string, any>; ledger_record?: any }> = []
  for (const work of protocol.requests) {
    const response = await fetch(`${base}/v1/work`, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(work) })
    const body = await response.json() as Record<string, any>
    const records = await (await fetch(`${base}/v1/executions`)).json() as { executions?: any[] }
    const ledger_record = (records.executions ?? []).find(row => row.request_id === body.request_id)
    results.push({ request: work, response: body, ...(ledger_record ? { ledger_record } : {}) })
  }
  const archive = ledger.read()
  const output = {
    schema_version: "stephanie.sc3-worker-run.v1", experiment_id: protocol.experiment_id,
    protocol_digest: protocol.protocol_digest, worker_source_commit: protocol.worker_head,
    worker_service: { host: server.hostname, port: server.port, transport_health: health, strategy_catalogue: strategies },
    fixture_provider: { kind: "deterministic scripted provider", actual_model_invocations: 0, scripted_provider_calls: provider.calls },
    storage: { configured_execution_ledger: ledgerPath, effective_execution_ledger: ledger.path,
      work_usage_ledger: "disabled", provider_usage_ledger: "disabled", evidence_ledger: "disabled", operational_storage_fallback: false,
      ledger_read: { record_count: archive.records.length, malformed: archive.malformed, conflicts: archive.conflicts, unavailable: archive.unavailable } },
    results,
  }
  mkdirSync(dirname(resolve(outputPath)), { recursive: true })
  writeFileSync(outputPath, JSON.stringify(output, null, 2) + "\n", "utf8")
  console.log(JSON.stringify({ status: "complete", experiment_id: protocol.experiment_id, results: results.length, ledger_records: archive.records.length, output_path: resolve(outputPath) }))
} finally {
  server.stop(true)
}
