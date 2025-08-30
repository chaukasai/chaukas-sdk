"""
Chaukas client implementation following chaukas-spec-client interface.
"""

import asyncio
import logging
import uuid
from typing import Optional, Dict, Any, List
from datetime import datetime, timezone

import httpx
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class ChaukasEvent(BaseModel):
    """Event model following chaukas-spec-client specification."""
    
    event_id: str
    session_id: str
    trace_id: str
    span_id: str
    parent_span_id: Optional[str] = None
    timestamp: datetime
    event_type: str
    source: str
    data: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None


class ChaukasClient:
    """
    Client for sending events to Chaukas platform.
    Implements the chaukas-spec-client interface.
    """
    
    def __init__(
        self,
        endpoint: str,
        api_key: str,
        timeout: float = 30.0,
        batch_size: int = 100,
        flush_interval: float = 5.0,
    ):
        self.endpoint = endpoint.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        
        self._events_queue: List[ChaukasEvent] = []
        self._client = httpx.AsyncClient(
            timeout=timeout,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "User-Agent": "chaukas-sdk/0.1.0",
            }
        )
        self._flush_task: Optional[asyncio.Task] = None
        self._closed = False
    
    async def send_event(self, event: ChaukasEvent) -> None:
        """Send a single event to the platform."""
        if self._closed:
            logger.warning("Client is closed, ignoring event")
            return
        
        self._events_queue.append(event)
        
        if len(self._events_queue) >= self.batch_size:
            await self._flush_events()
    
    async def send_events(self, events: List[ChaukasEvent]) -> None:
        """Send multiple events in batch."""
        if self._closed:
            logger.warning("Client is closed, ignoring events")
            return
        
        self._events_queue.extend(events)
        
        if len(self._events_queue) >= self.batch_size:
            await self._flush_events()
    
    async def _flush_events(self) -> None:
        """Flush queued events to the platform."""
        if not self._events_queue:
            return
        
        events_to_send = self._events_queue[:]
        self._events_queue.clear()
        
        try:
            payload = {
                "events": [event.model_dump() for event in events_to_send]
            }
            
            response = await self._client.post(
                f"{self.endpoint}/events",
                json=payload
            )
            response.raise_for_status()
            
            logger.debug(f"Successfully sent {len(events_to_send)} events")
            
        except Exception as e:
            logger.error(f"Failed to send events: {e}")
            # Re-queue events for retry (simple strategy)
            self._events_queue = events_to_send + self._events_queue
    
    def create_event(
        self,
        event_type: str,
        source: str,
        data: Dict[str, Any],
        session_id: str,
        trace_id: str,
        span_id: str,
        parent_span_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ChaukasEvent:
        """Create a new event with the given parameters."""
        return ChaukasEvent(
            event_id=str(uuid.uuid4()),
            session_id=session_id,
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
            timestamp=datetime.now(timezone.utc),
            event_type=event_type,
            source=source,
            data=data,
            metadata=metadata,
        )
    
    async def flush(self) -> None:
        """Manually flush all queued events."""
        await self._flush_events()
    
    async def close(self) -> None:
        """Close the client and flush remaining events."""
        if self._closed:
            return
        
        self._closed = True
        
        if self._flush_task:
            self._flush_task.cancel()
        
        await self._flush_events()
        await self._client.aclose()
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()