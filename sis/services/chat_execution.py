"""One synchronous, bounded SIS conversation turn using existing services."""

from __future__ import annotations

import logging
import re
import asyncio
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, AsyncIterator

from stephanie.utils.secret_utils import redact_secrets
from sis.services.chat_context import build_chat_messages


_REQUEST_KEY = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")


class ConversationNotFound(Exception):
    pass


class ConversationBusy(Exception):
    pass


class IdempotencyConflict(Exception):
    pass


class ChatServiceUnavailable(Exception):
    pass


class ChatInferenceFailed(Exception):
    pass


class ConversationLocks:
    """Serialize same-process turns per conversation without holding DB transactions."""

    def __init__(self):
        self._guard = asyncio.Lock()
        self._entries: dict[int, tuple[asyncio.Lock, int]] = {}

    async def _drop_reference(self, conversation_id: int) -> None:
        async with self._guard:
            current = self._entries.get(conversation_id)
            if current is None:
                return
            current_lock, users = current
            if users <= 1:
                self._entries.pop(conversation_id, None)
            else:
                self._entries[conversation_id] = (current_lock, users - 1)

    @asynccontextmanager
    async def hold(self, conversation_id: int, timeout: float = 30.0) -> AsyncIterator[None]:
        async with self._guard:
            lock, users = self._entries.get(conversation_id, (asyncio.Lock(), 0))
            self._entries[conversation_id] = (lock, users + 1)
        try:
            await asyncio.wait_for(lock.acquire(), timeout=timeout)
        except asyncio.TimeoutError as exc:
            await self._drop_reference(conversation_id)
            raise ConversationBusy("A response is already being generated for this conversation") from exc
        except BaseException:
            await self._drop_reference(conversation_id)
            raise
        try:
            yield
        finally:
            lock.release()
            await self._drop_reference(conversation_id)


class ConversationExecutionService:
    def __init__(self, *, memory, container, cfg, logger=None, locks=None):
        self.memory = memory
        self.container = container
        self.cfg = cfg
        self.logger = logger or logging.getLogger(__name__)
        self.locks = locks or ConversationLocks()
        self._active_guard = asyncio.Lock()
        self._active: dict[tuple[int, str], asyncio.Task] = {}
        self._cancelled: set[tuple[int, str]] = set()

    @staticmethod
    def _validate_request_id(request_id: str) -> str:
        if not _REQUEST_KEY.fullmatch(request_id or ""):
            raise ValueError("Idempotency-Key must contain 1–128 safe characters")
        return request_id

    async def create_conversation(
        self, *, username: str, title: str, request_id: str
    ):
        request_id = self._validate_request_id(request_id)
        async with self.locks.hold(-1):
            return self.memory.chats.add_conversation_once(
                {
                    "provider": "stephanie",
                    "title": title,
                    "meta": {"source": "sis_live", "sis_owner": username},
                },
                request_id,
            )

    async def execute_turn(
        self,
        *,
        conversation_id: int,
        username: str,
        text: str,
        request_id: str,
    ) -> dict[str, Any]:
        request_id = self._validate_request_id(request_id)
        settings = self.cfg.get("chat_runtime", {})
        store = self.memory.chats

        async with self.locks.hold(conversation_id):
            conversation = store.get_conversation(conversation_id)
            if conversation is None:
                raise ConversationNotFound()
            owner = (conversation.meta or {}).get("sis_owner")
            # The current SIS account is deliberately single-user. Legacy
            # records with no owner remain readable by that sole account.
            if owner and owner != username:
                raise ConversationNotFound()

            prior_user = store.get_message_by_request_id(
                conversation_id, request_id, role="user"
            )
            if prior_user is not None and prior_user.text != text:
                raise IdempotencyConflict()
            prior_answer = store.get_message_by_request_id(
                conversation_id, request_id, role="assistant"
            )
            if prior_answer is not None:
                turn = next(
                    (
                        item
                        for item in store.get_turns_for_conversation(conversation_id)
                        if item.assistant_message_id == prior_answer.id
                    ),
                    None,
                )
                if turn is None:
                    raise RuntimeError("Persisted response is missing its ChatTurn")
                return {
                    "conversation_id": conversation_id,
                    "user_message": prior_user.to_dict(),
                    "assistant_message": prior_answer.to_dict(),
                    "turn": turn.to_dict(include_messages=False),
                    "model": (prior_answer.meta or {}).get("model"),
                    "replayed": True,
                }

            if prior_user is not None and (prior_user.meta or {}).get("execution_status") == "in_progress":
                created_at = getattr(prior_user, "created_at", None)
                age_seconds = (
                    (datetime.now(created_at.tzinfo) - created_at).total_seconds()
                    if isinstance(created_at, datetime)
                    else 0
                )
                retry_after = max(180, int(settings.get("model", {}).get("timeout", 60)) + 60)
                if age_seconds < retry_after:
                    raise ConversationBusy("This request is already being processed")

            try:
                llm = self.container.get("llm")
            except Exception as exc:
                self.logger.log(
                    "SISChatServiceUnavailable",
                    {"conversation_id": conversation_id, "error_type": type(exc).__name__},
                )
                raise ChatServiceUnavailable() from exc

            history = store.get_messages_before(
                conversation_id,
                prior_user.order_index if prior_user else 2**63 - 1,
                limit=max(0, int(settings.get("history_limit", 24))),
            )
            try:
                messages = build_chat_messages(
                    history,
                    text,
                    settings,
                )
            except ValueError:
                raise

            if prior_user is None:
                try:
                    user_message = store.add_user_message_once(
                        conversation_id,
                        request_id,
                        text,
                        meta={"source": "sis_live", "sis_owner": username},
                    )
                except ValueError as exc:
                    if "Idempotency key" in str(exc):
                        raise IdempotencyConflict() from exc
                    raise
            else:
                user_message = prior_user
                store.mark_user_message_status(
                    conversation_id, user_message.id, "in_progress"
                )

            model_cfg = settings.get("model", {}) or {}
            agent_cfg = {
                "name": "sis_chat",
                "model": dict(model_cfg),
                "use_memory_for_fast_prompts": False,
            }
            active_key = (conversation_id, request_id)
            current_task = asyncio.current_task()
            async with self._active_guard:
                self._cancelled.discard(active_key)
                self._active[active_key] = current_task
            try:
                if hasattr(llm, "achat"):
                    result = await llm.achat(
                        messages,
                        context={"conversation_id": conversation_id},
                        agent_cfg=agent_cfg,
                        agent_name="sis_chat",
                        add_to_history=False,
                    )
                else:
                    result = llm.chat(
                        messages,
                        context={"conversation_id": conversation_id},
                        agent_cfg=agent_cfg,
                        agent_name="sis_chat",
                        add_to_history=False,
                    )
                    if hasattr(result, "__await__"):
                        result = await result
                if not isinstance(result, dict):
                    raise TypeError("LLMService.chat returned an invalid result")
                answer = str(result.get("text") or "").strip()
                if not answer:
                    raise ValueError("LLMService.chat returned an empty answer")
            except asyncio.CancelledError:
                try:
                    store.mark_user_message_status(
                        conversation_id,
                        user_message.id,
                        "cancelled",
                        error_type="CancelledError",
                    )
                except Exception as persist_exc:
                    self.logger.log(
                        "SISChatCancellationPersistFailed",
                        {"conversation_id": conversation_id, "error_type": type(persist_exc).__name__},
                    )
                self.logger.log(
                    "SISChatInferenceCancelled",
                    {"conversation_id": conversation_id, "request_id": request_id},
                )
                raise
            except Exception as exc:
                try:
                    store.mark_user_message_status(
                        conversation_id,
                        user_message.id,
                        "failed",
                        error_type=type(exc).__name__,
                    )
                except Exception as persist_exc:
                    self.logger.log(
                        "SISChatFailureStatePersistFailed",
                        {"conversation_id": conversation_id, "error_type": type(persist_exc).__name__},
                    )
                self.logger.log(
                    "SISChatInferenceFailed",
                    {
                        "conversation_id": conversation_id,
                        "request_id": request_id,
                        "error_type": type(exc).__name__,
                    },
                )
                raise ChatInferenceFailed() from exc
            finally:
                async with self._active_guard:
                    if self._active.get(active_key) is current_task:
                        self._active.pop(active_key, None)

            provenance = {
                "runtime": "stephanie-llm",
                "provider": result.get("provider"),
                "model": result.get("model_name"),
                "cached": bool(result.get("cached", False)),
                "response_id": result.get("response_id"),
                "usage": result.get("usage"),
                "latency_ms": result.get("latency_ms"),
                "finish_reason": result.get("finish_reason"),
            }
            provenance = {key: value for key, value in provenance.items() if value is not None}
            # Redact as a final guard before persisting provider metadata.
            provenance = redact_secrets(provenance)
            try:
                assistant, turn = store.complete_assistant_turn(
                    conversation_id,
                    user_message.id,
                    request_id,
                    answer,
                    meta=provenance,
                )
            except Exception as exc:
                # Do not return an answer as successful when the canonical
                # transcript/turn could not be committed atomically.
                try:
                    store.mark_user_message_status(
                        conversation_id,
                        user_message.id,
                        "failed",
                        error_type=type(exc).__name__,
                    )
                except Exception as persist_exc:
                    self.logger.log(
                        "SISChatFailureStatePersistFailed",
                        {"conversation_id": conversation_id, "error_type": type(persist_exc).__name__},
                    )
                self.logger.log(
                    "SISChatPersistenceFailed",
                    {"conversation_id": conversation_id, "error_type": type(exc).__name__},
                )
                raise ChatInferenceFailed() from exc

            user_message.meta = {
                **(user_message.meta or {}),
                "execution_status": "completed",
            }
            user_message.meta.pop("error_type", None)
            return {
                "conversation_id": conversation_id,
                "user_message": user_message.to_dict(),
                "assistant_message": assistant.to_dict(),
                "turn": turn.to_dict(include_messages=False),
                "model": result.get("model_name"),
                "provider": result.get("provider"),
                "replayed": False,
            }

    async def cancel_turn(
        self, *, conversation_id: int, username: str, request_id: str
    ) -> bool:
        """Best-effort cancellation of an active async provider call."""
        self._validate_request_id(request_id)
        conversation = self.memory.chats.get_conversation(conversation_id)
        if conversation is None:
            raise ConversationNotFound()
        owner = (conversation.meta or {}).get("sis_owner")
        if owner and owner != username:
            raise ConversationNotFound()
        async with self._active_guard:
            key = (conversation_id, request_id)
            task = self._active.get(key)
            if task is None or task.done():
                return False
            self._cancelled.add(key)
            task.cancel()
            return True

    async def consume_cancelled(self, conversation_id: int, request_id: str) -> bool:
        async with self._active_guard:
            key = (conversation_id, request_id)
            if key not in self._cancelled:
                return False
            self._cancelled.remove(key)
            return True
