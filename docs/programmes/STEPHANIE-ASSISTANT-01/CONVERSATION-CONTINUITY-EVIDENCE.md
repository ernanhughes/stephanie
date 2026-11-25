# Conversation continuity evidence

The real qualification used ChatStore conversation ID `1` in `stephanie_assistant_01`:

1. Synthetic objective: prepare three study notes by Friday.
2. Stephanie acknowledged and preserved it.
3. A follow-up asked what needed finishing; Stephanie answered “three study notes by Friday.”

Both responses were recorded as `opencode-go/deepseek-v4.1-flash` with provider usage. A new Python/SIS process reopened the isolated database and observed 4 messages, 2 turns, and the same two model identities. A POST replay using the same idempotency key returned the stored answer with `replayed=true`; it did not invoke another model.

The deterministic test `test_one_conversation_continues_after_model_change_without_losing_objective` uses two fixture models and checks the same conversation ID, retained objective/answer and follow-up context. It passed. This test verifies functional continuation after a model switch; no live switch was attempted because the local model response was not qualified and the six-call ceiling was reached.
