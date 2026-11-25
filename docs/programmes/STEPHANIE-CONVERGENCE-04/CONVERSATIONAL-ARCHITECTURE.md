# Conversational architecture

## Ownership map

- **Stephanie:** conversational identity, dialog behavior, task framing, retrieved context selection, investigation/candidate coordination, recommendations and retained experience.
- **AI (`C:\Projects\ai`):** provider-neutral inference contracts, adapters, structured output, budgets and telemetry.
- **Worker:** bounded agent/model/tool execution and execution evidence; no approval authority.
- **Praxis:** comparative effectiveness evaluation; no Mission acceptance authority.
- **Mission Control:** project/objective context, Work Envelopes, human authorization, operational state and acceptance.
- **ECK and existing specialist applications:** reuse their relevant action-selection and domain mechanisms; do not duplicate them inside Stephanie.

## Continuity and memory

Preserve conversation identity across inference route changes by selecting relevant chat history and explicit context records. Keep user preferences, project facts, active objective, decision history and reviewed experience distinct and source-linked. ChatStore is conversational history; CaseBook/MemCube and shared AI memory are retrieval/experience mechanisms, not a substitute for source systems. Mission remains authoritative for objective and operational status. A model provider session is transient execution state.

The exact SIS ChatStore persistence backend, memory ownership boundaries, project context adapter, ECK integration, and cross-model compaction behavior require source-level qualification before they are called durable personal memory. Current SC4 does not wire Stephanie into Mission chat.

## Model independence and delegation

Use shared `ai.inference` contracts for internal model selection, recording the requested/effective provider route and usage. Agent/application delegation is a separate operation and must use Mission authority and existing Worker/application interfaces, preserving identity and evidence. Never treat a model choice as delegation or an inference response as an authorized action.

## Roadmap boundary

The current SC4 Mission integration is not a general chat interface. Project-aware conversation, persistent preference controls, route switching with continuity, specialist delegation and unified work management remain roadmap requirements. Build them by extending existing stores and interfaces after source ownership and retention rules are verified.
