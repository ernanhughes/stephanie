# Worker adapter evidence

Read-only routes are `GET /v1/strategies` and `GET /v1/executions`; Worker source exposes a maximum 100-record recent read window, gated to loopback and an available execution ledger. The adapter never calls a work or mutation route. Empty successful windows are `inconclusive`; missing ledger/service is unavailable or missing.

Live check at `http://127.0.0.1:7317` was refused, so no current Worker records were imported. Existing SE2Q summarized evidence remains the basis for the investigation: 2 dispatched empty results (execution unknown), one malformed at 512-token ceiling, one usable 1024-token follow-up, one pre-dispatch block. This is historical, not current reliability proof.
