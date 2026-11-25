# Talking to Stephanie

## Start

From PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File C:\Projects\stephanie\scripts\start_stephanie_assistant_01.ps1
```

Open [http://127.0.0.1:8766/chats](http://127.0.0.1:8766/chats). The service binds to loopback and stores only this release's transcript in PostgreSQL database `stephanie_assistant_01`. Stop it with **Ctrl+C** in the serving PowerShell window.

The repository-root `.env` supplies the configured PostgreSQL password and OpenCode key; keep it private. Do not copy either credential into chat. OpenCode Go requests require the external-approval checkbox. Leave it unchecked for local-only turns. Any selected/retrieved Mission project context stays local even when external approval is checked. Legacy/private earlier turns also prevent external routing.

Use the Mission dropdown to attach read-only current project evidence. Responses show their effective provider/model, data class, route state, Mission source and observation time. A same-key retry replays the stored answer. Local-only inference uses `ollama/mistral-small:latest`; its response quality has not been qualified and may be limited.

OpenCode Free is not the active direct API path: its CLI probe reached the usage limit, and current provider docs prohibit third-party API access for Free chat models. Go DeepSeek is the only configured external model and uses existing subscription allowance. No pay-as-you-go model or automatic paid fallback is configured.
