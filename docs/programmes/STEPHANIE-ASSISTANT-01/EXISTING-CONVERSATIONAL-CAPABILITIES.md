# Existing conversational capabilities

| Capability | Existing implementation | SA1 use |
|---|---|---|
| Browser interface | SIS server-rendered `sis/templates/chats/{list,detail}.html` | Extended in place; no frontend or SPA added |
| Transcript and identity | `ChatStore`, `ChatConversationORM`, `ChatMessageORM`, `ChatTurnORM` | Canonical conversation identity and persistence |
| Turn execution | `sis/services/chat_execution.py` | Bounded history, same-conversation lock, idempotency, failure state |
| Model invocation | `stephanie/services/llm_service.py`, LiteLLM `achat` | Reused; route policy supplies eligible OpenCode Go requests |
| Project evidence | Mission Control HTTP API | New small GET-only projection client; no direct DB access |
| Memory and old conversations | Existing MemCube/CaseBook/ChatStore capabilities | Not automatically searched by this release; only bounded current-conversation history is sent |
| Worker/Praxis/improvement | Existing governed interfaces | Preserved and not invoked by chat |

Chat stays off in `config/sis.yaml`. The separate `sis_assistant_01` config uses an isolated database and only turns on after the observed model request and cross-process ChatStore persistence check. The flags remain explicit assertions, not probes.
