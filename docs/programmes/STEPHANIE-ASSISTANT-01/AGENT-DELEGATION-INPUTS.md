# Future agent delegation inputs

Delegation is future work and is not enabled by SA1. Keep three concepts separate:

1. **Model substitution:** another provider answers the same Stephanie conversation turn. Preserve the ChatStore ID, context policy, effective model and usage evidence.
2. **Specialist delegation:** a bounded specialist receives a task-specific brief and returns attributed evidence. It is not a replacement Stephanie identity.
3. **Governed execution:** Worker/Praxis/Mission capabilities require their existing explicit authorization and review workflow; a chat message alone does not commission them.

Candidates: Codex and Claude only through supported APIs/runtime access; ChatGPT/GPT only through an available approved API; DeepSeek/OpenCode Go; Muse only through an actually configured supported route; existing Worker, Praxis, Writer, Publication, Inspect and AITEST interfaces. Consumer subscriptions do not imply API credits or access. Never assume the assistant can read private consumer chat histories. Future work must define recipient data classes, budget/call ceilings, idempotency, cancellation, provenance, approval boundary and human review before enabling delegation.
