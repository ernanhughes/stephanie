# Conversational data policy

The decision is deterministic and applies to the context SIS assembles for the turn: bounded user/assistant history, current text, and any retrieved Mission projection.

| Class | Behavior |
|---|---|
| `approved_external` | OpenCode Go is eligible only when the current turn checkbox is checked and every included historical message has that label |
| `private_project` | Local route only; selected or automatically retrieved Mission data always forces this class |
| `unknown` | Local route only; this is the default and includes legacy messages without provenance |
| credentials | Reject before persisting/inference when deterministic credential patterns match |
| protected evaluation | Reject before persisting/inference when protected-evaluation markers match |

The UI approval text explicitly covers the current turn and assembled history. Mission project context has its own display notice and always remains local in this release. Retrieved projection text is scanned for credential and protected-evaluation patterns before model invocation. Any classification uncertainty stays local. Local quality is not independently qualified; failures are retained as safe state/type evidence and no external escalation occurs.

Not implemented: attachment ingestion, arbitrary tool-result retrieval, global preference retrieval, or cross-conversation memory search. Those sources cannot silently enter a prompt.
