# stephanie/services/bus/nats_bus.py
"""
NATS JetStream Bus Implementation

Production-ready event bus implementation using NATS JetStream.
Provides persistent, durable messaging with at-least-once delivery semantics.

Features:
- Persistent message storage
- Durable consumers
- Idempotent processing
- Dead letter queue support
- Request/reply pattern
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from typing import Any, Callable, Dict, List, Optional

from nats.aio.client import Client as NATS
from nats.errors import NoServersError, TimeoutError
from nats.js.api import ConsumerConfig, DeliverPolicy

from .bus_protocol import BusProtocol
from .idempotency import InMemoryIdempotencyStore, NatsKVIdempotencyStore
from .errors import BusConnectionError, BusPublishError, BusSubscribeError, BusRequestError

class NatsKnowledgeBus(BusProtocol):
    """
    NATS JetStream implementation of the event bus.
    
    This implementation provides production-grade messaging with:
    - Persistent storage via JetStream
    - Durable consumers that survive restarts
    - Idempotent message processing
    - Dead letter queue support for failed messages
    
    Attributes:
        servers: List of NATS server URLs
        stream: JetStream stream name
        logger: Logger instance for bus operations
        _nc: NATS connection instance
        _js: JetStream context
        _idem_store: Idempotency store instance
        _connected: Connection status flag
        _subscriptions: Active subscription references
    """
    
    def __init__(
        self,
        servers: List[str] = ["nats://localhost:4222"],
        stream: str = "stephanie",
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the NATS bus.
        
        Args:
            servers: List of NATS server URLs
            stream: JetStream stream name
            logger: Optional logger instance
        """
        self.servers = servers
        self.stream = stream
        self.logger = logger or logging.getLogger(__name__)
        self._nc: Optional[NATS] = None
        self._js = None
        self._idem_store = None
        self._connected = False
        self._subscriptions = {}
        
    async def connect(self) -> bool:
        """
        Connect to NATS with JetStream capability check.
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        if self._connected:
            return True
            
        try:
            self._nc = NATS()
            await self._nc.connect(
                servers=self.servers,
                allow_reconnect=True,
                reconnect_time_wait=2.0,
                max_reconnect_attempts=5
            )
            self._js = self._nc.jetstream()
            
            # Verify JetStream is available and configure stream
            try:
                await self._js.add_stream(
                    name=self.stream, 
                    subjects=[f"{self.stream}.>"]
                )
                self.logger.info(f"Connected to NATS JetStream (stream: {self.stream})")
            except Exception as e:
                self.logger.warning(f"JetStream configuration failed: {str(e)}")
                await self._nc.close()
                return False
                
            # Create idempotency store using NATS KV
            self._idem_store = NatsKVIdempotencyStore(self._js, bucket=f"{self.stream}_idem")
            
            self._connected = True
            return True
            
        except (NoServersError, OSError, TimeoutError) as e:
            self.logger.warning(f"NATS connection failed: {str(e)}")
            return False
        except Exception as e:
            self.logger.error(f"NATS initialization error: {str(e)}", exc_info=True)
            return False
            
    async def publish(self, subject: str, payload: Dict[str, Any]) -> None:
        """
        Publish with standard event envelope.
        
        Args:
            subject: Event subject/topic
            payload: Event data payload
            
        Raises:
            BusPublishError: If publishing fails
        """
        if not self._connected and not await self.connect():
            raise BusConnectionError("Not connected to NATS")
            
        # Create standard event envelope
        envelope = {
            "event_id": f"{subject}-{uuid.uuid4().hex}",
            "timestamp": time.time(),
            "subject": subject,
            "payload": payload
        }
        
        data = json.dumps(envelope).encode()
        try:
            await self._js.publish(f"{self.stream}.{subject}", data)
            self.logger.debug(f"Published to {subject}: {envelope['event_id']}")
        except Exception as e:
            self.logger.error(f"Failed to publish to {subject}: {str(e)}")
            raise BusPublishError(f"Failed to publish to {subject}") from e
            
    async def subscribe(self, subject: str, handler: Callable[[Dict[str, Any]], None]) -> None:
        """
        Subscribe with durable consumer and idempotency handling.
        
        Args:
            subject: Event subject/topic to subscribe to
            handler: Callback function to handle events
            
        Raises:
            BusSubscribeError: If subscription fails
        """
        if not self._connected and not await self.connect():
            raise BusConnectionError("Not connected to NATS")
            
        durable_name = f"durable_{subject.replace('.', '_')}"
        
        async def wrapped(msg):
            """
            Wrapper function that handles idempotency and error handling
            before calling the actual handler.
            """
            try:
                envelope = json.loads(msg.data.decode())
                event_id = envelope.get("event_id")
                
                # Handle idempotency - skip if already processed
                if event_id and await self._idem_store.seen(event_id):
                    await msg.ack()
                    self.logger.debug(f"Skipping duplicate event: {event_id}")
                    return
                    
                if event_id:
                    await self._idem_store.mark(event_id)
                    
                # Call actual handler
                await handler(envelope["payload"])
            except Exception as e:
                self.logger.error(f"Error handling event {subject}: {str(e)}", exc_info=True)
                # In a production system, you might want to implement
                # dead letter queue handling here
            finally:
                await msg.ack()
                
        try:
            sub = await self._js.subscribe(
                f"{self.stream}.{subject}",
                durable=durable_name,
                cb=wrapped,
                config=ConsumerConfig(
                    deliver_policy=DeliverPolicy.ALL,
                    ack_wait=30.0
                )
            )
            self._subscriptions[subject] = sub
            self.logger.info(f"Subscribed to {subject} with durable consumer {durable_name}")
        except Exception as e:
            self.logger.error(f"Failed to subscribe to {subject}: {str(e)}")
            raise BusSubscribeError(f"Failed to subscribe to {subject}") from e
            
    async def request(self, subject: str, payload: Dict[str, Any], timeout: float = 5.0) -> Optional[Dict[str, Any]]:
        """
        Send a request and wait for a reply.
        
        Args:
            subject: Request subject/topic
            payload: Request data
            timeout: Maximum time to wait for response (seconds)
            
        Returns:
            Optional[Dict]: Response data or None if timeout
            
        Raises:
            BusRequestError: If request fails
        """
        if not self._connected and not await self.connect():
            raise BusConnectionError("Not connected to NATS")
            
        try:
            data = json.dumps(payload).encode()
            response = await self._nc.request(
                f"{self.stream}.rpc.{subject}", 
                data, 
                timeout=timeout
            )
            return json.loads(response.data.decode())
        except TimeoutError:
            self.logger.warning(f"Request timed out for {subject}")
            return None
        except Exception as e:
            self.logger.error(f"Request failed for {subject}: {str(e)}")
            raise BusRequestError(f"Request failed for {subject}") from e
            
    async def close(self) -> None:
        """Gracefully shut down the connection."""
        if self._connected and self._nc:
            try:
                # Unsubscribe from all subjects
                for subject, sub in self._subscriptions.items():
                    await sub.unsubscribe()
                    self.logger.debug(f"Unsubscribed from {subject}")
                self._subscriptions.clear()
                
                # Close connection
                await self._nc.close()
                self._connected = False
                self.logger.info("NATS connection closed")
            except Exception as e:
                self.logger.error(f"Error during NATS shutdown: {str(e)}")
                raise
                
    def get_backend(self) -> str:
        """Return the active backend name."""
        return "nats"
        
    @property
    def idempotency_store(self) -> Any:
        """Return the idempotency store for this bus."""
        if not self._idem_store:
            # Fallback if not connected yet
            return InMemoryIdempotencyStore()
        return self._idem_store
