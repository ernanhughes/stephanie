# stephanie/services/bus/nats_bus.py
"""
NATS JetStream implementation of the BusProtocol.
Handles persistent messaging with idempotency and DLQ support.
"""

from __future__ import annotations

import asyncio
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


class NatsKnowledgeBus(BusProtocol):
    """NATS JetStream implementation of the event bus."""
    
    def __init__(
        self,
        servers: List[str] = ["nats://localhost:4222"],
        stream: str = "stephanie",
        logger: Optional[logging.Logger] = None
    ):
        self.servers = servers
        self.stream = stream
        self.logger = logger or logging.getLogger(__name__)
        self._nc: Optional[NATS] = None
        self._js = None
        self._idem_store = None
        self._connected = False
        self._subscriptions = {}
        
    async def connect(self) -> bool:
        """Connect to NATS with JetStream capability check."""
        if self._connected:
            return True
            
        try:
            self._nc = NATS()
            await self._nc.connect(
                servers=self.servers,
                allow_reconnect=True,
                reconnect_time_wait=2.0
            )
            self._js = self._nc.jetstream()
            
            # Verify JetStream is available
            try:
                await self._js.add_stream(name=self.stream, subjects=[f"{self.stream}.>"])
                self.logger.info(f"Connected to NATS JetStream (stream: {self.stream})")
            except Exception as e:
                self.logger.warning(f"JetStream not available: {str(e)}")
                await self._nc.close()
                return False
                
            # Create idempotency store
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
        """Publish with standard event envelope."""
        if not self._connected and not await self.connect():
            return
            
        envelope = {
            "event_id": f"{subject}-{uuid.uuid4().hex}",
            "timestamp": time.time(),
            "subject": subject,
            "payload": payload
        }
        
        data = json.dumps(envelope).encode()
        try:
            await self._js.publish(f"{self.stream}.{subject}", data)
        except Exception as e:
            self.logger.error(f"Failed to publish to {subject}: {str(e)}")
            
    async def subscribe(self, subject: str, handler: Callable[[Dict[str, Any]], None]) -> None:
        """Subscribe with durable consumer and idempotency handling."""
        if not self._connected and not await self.connect():
            return
            
        durable_name = f"durable_{subject.replace('.', '_')}"
        
        async def wrapped(msg):
            try:
                envelope = json.loads(msg.data.decode())
                event_id = envelope.get("event_id")
                
                # Handle idempotency
                if event_id and await self._idem_store.seen(event_id):
                    await msg.ack()
                    return
                    
                if event_id:
                    await self._idem_store.mark(event_id)
                    
                # Call actual handler
                await handler(envelope["payload"])
            except Exception as e:
                self.logger.error(f"Error handling event {subject}: {str(e)}", exc_info=True)
                # DLQ handling would go here
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
            
    async def request(self, subject: str, payload: Dict[str, Any], timeout: float = 5.0) -> Optional[Dict[str, Any]]:
        """Send a request and wait for a reply."""
        if not self._connected and not await self.connect():
            return None
            
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
            return None
            
    async def close(self) -> None:
        """Gracefully shut down the connection."""
        if self._connected and self._nc:
            try:
                # Unsubscribe from all subjects
                for sub in self._subscriptions.values():
                    await sub.unsubscribe()
                self._subscriptions.clear()
                
                # Close connection
                await self._nc.close()
                self._connected = False
                self.logger.info("NATS connection closed")
            except Exception as e:
                self.logger.error(f"Error during NATS shutdown: {str(e)}")
                
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