/** Bounded genuine local-model probe through the isolated Worker HTTP API. */
import { readFileSync, writeFileSync, mkdirSync } from "node:fs"
import { dirname, resolve } from "node:path"
import { createWorkerServer } from "../../../worker/src/server/server.js"
import { WorkerService } from "../../../worker/src/core/worker.js"
import { ExecutionLedger } from "../../../worker/src/core/execution-ledger.js"
import { ModelRuntime } from "../../../worker/src/models/runtime/engine.js"
import { loadModelsConfig } from "../../../worker/src/models/runtime/config.js"
import { ModelExecutor } from "../../../worker/src/executors/model.js"
import { ExecutorRegistry } from "../../../worker/src/executors/types.js"

const root = resolve(process.argv[2] ?? "")
const outputPath = resolve(process.argv[3] ?? "")
if (!process.argv[2] || !process.argv[3]) throw new Error("usage: bun sc3_worker_live_runner.ts <isolated-root> <result.json>")
mkdirSync(root, { recursive: true })
const ledgerPath = resolve(root, "worker-live-execution-ledger.jsonl")
const modelsPath = process.env.WORKER_MODELS_CONFIG
if (!modelsPath) throw new Error("WORKER_MODELS_CONFIG must point to the inspected Worker model config")
const configBytes = readFileSync(modelsPath)
const models = loadModelsConfig(modelsPath)
const policy = models.policies["h4.local"]
if (!policy || policy.required_data_policy !== "local_only" || policy.routes.join(",") !== "h4_qwen_local" || models.models.h4_qwen_local?.data_policy !== "local_only") {
  throw new Error("local readiness route does not match the inspected h4.local/qwen local-only configuration")
}
const runtime = new ModelRuntime(models)
const registry = new ExecutorRegistry()
registry.register(new ModelExecutor(runtime))
const ledger = new ExecutionLedger(ledgerPath, true)
const service = new WorkerService(registry, "model", {}, runtime, undefined, undefined, undefined, undefined, undefined, ledger)
const server = createWorkerServer(service, { host: "127.0.0.1", port: 0 })
const base = `http://${server.hostname}:${server.port}`
const schema = { type: "object", properties: { answer: { type: "string" } }, required: ["answer"], additionalProperties: false }
const output: any = { schema_version: "stephanie.sc3-live-probe.v1", route: "h4.local/h4_qwen_local", policy: "local_only",
  model_config_sha256: await crypto.subtle.digest("SHA-256", configBytes).then(data => Buffer.from(data).toString("hex")),
  isolated_storage: { execution_ledger: ledgerPath, other_worker_ledgers: "disabled", operational_fallback: false },
  actual_model_invocations_maximum: 4, attempts: [], stopped_after_readiness_failure: false }
function request(taskId: string, strategy: "structured.v1" | "adaptive.v1") {
  return { schema: "worker.work-request.v1", app: "stephanie-sc3-live", task_id: taskId,
    objective: taskId === "SC3-LIVE-READINESS"
      ? 'Reply with exactly one JSON object: {"answer":"ready"}. No prose.'
      : 'Reply with exactly one JSON object: {"answer":"correct"}. No prose.',
    execution: { strategy_id: strategy, ...(strategy === "adaptive.v1" ? { normalization_policy: "json-fence.v1" } : {}), model: "h4_qwen_local", policy: "h4.local" },
    parameters: { data_policy: "local_only", timeout_ms: 30_000, max_output_tokens: 128, temperature: 0,
      output_schema: schema }, options: { acceptance: "human" }, metadata: { operation_id: "sc3-genuine-local-probe", stage_id: taskId } }
}
async function execute(taskId: string, strategy: "structured.v1" | "adaptive.v1", expected: string) {
  const work = request(taskId, strategy)
  const response = await fetch(`${base}/v1/work`, { method: "POST", headers: { "content-type": "application/json" },
    body: JSON.stringify(work), signal: AbortSignal.timeout(45_000) })
  const result = await response.json() as Record<string, any>
  const archive = await (await fetch(`${base}/v1/executions`)).json() as { executions?: any[] }
  const ledgerRow = (archive.executions ?? []).find(row => row.request_id === result.request_id)
  const outputText = typeof result.output === "string" ? result.output : ""
  let parsed: any = null
  try { parsed = JSON.parse(outputText) } catch {}
  const row = { task_id: taskId, strategy_id: strategy, request: work, result,
    ledger_record: ledgerRow ?? null, oracle: { expected_value_not_persisted: true, match: parsed?.answer === expected },
    usage: (result.strategy_execution?.resources ?? null), status: result.status }
  output.attempts.push(row)
  writeFileSync(outputPath, JSON.stringify(output, null, 2) + "\n", "utf8")
  return row
}
try {
  const health = await (await fetch(`${base}/health/transport`)).json()
  output.worker = { host: server.hostname, port: server.port, health,
    strategy_catalogue: await (await fetch(`${base}/v1/strategies`)).json() }
  const ready = await execute("SC3-LIVE-READINESS", "structured.v1", "ready")
  if (ready.status !== "AWAITING_ACCEPTANCE" || ready.oracle.match !== true || !ready.ledger_record) {
    output.stopped_after_readiness_failure = true
    output.stop_reason = "local model route did not produce a ledger-backed output matching the readiness oracle; comparison lane not started"
  } else {
    await execute("SC3-LIVE-BASELINE", "structured.v1", "correct")
    await execute("SC3-LIVE-CANDIDATE", "adaptive.v1", "correct")
  }
  output.model_invocation_observation = output.attempts.map((attempt: any) => ({
    task_id: attempt.task_id, status: attempt.status,
    invocation_count: attempt.usage?.invocation_count ?? null,
    observed_model: attempt.result?.strategy_execution?.invocations?.map((item: any) => item.observed_model).filter(Boolean) ?? [],
    output_tokens: attempt.usage?.observed_output_tokens ?? null,
    cost_usd: null, cost_status: "unknown",
  }))
  output.execution_ledger = ledger.read()
  output.worker_model_config_path = modelsPath
  output.actual_model_invocations = output.attempts.reduce((sum: number, item: any) => sum + (item.usage?.invocation_count ?? 0), 0)
  mkdirSync(dirname(outputPath), { recursive: true })
  writeFileSync(outputPath, JSON.stringify(output, null, 2) + "\n", "utf8")
  console.log(JSON.stringify({ status: output.stopped_after_readiness_failure ? "stopped_at_readiness" : "complete",
    attempts: output.attempts.length, actual_model_invocations: output.actual_model_invocations, output_path: outputPath }))
} finally { server.stop(true) }
