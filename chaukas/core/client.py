"""
Chaukas client implementation using proto messages for 100% spec compliance.
"""

import asyncio
import logging
from typing import Optional, List, Union

import httpx

from chaukas.spec.common.v1.events_pb2 import Event, EventBatch
from chaukas.spec.client.v1.client_pb2 import IngestEventRequest, IngestEventBatchRequest

from .config import ChaukasConfig, get_config
from .proto_wrapper import EventWrapper

logger = logging.getLogger(__name__)


class ChaukasClient:
    """
    Client for sending events to Chaukas platform.
    Uses proto messages for 100% spec compliance.
    """
    
    def __init__(
        self,
        config: Optional[ChaukasConfig] = None,
        endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: Optional[float] = None,
        batch_size: Optional[int] = None,
        flush_interval: Optional[float] = None,
    ):
        """
        Initialize Chaukas client.
        
        Args:
            config: ChaukasConfig instance (uses env config if not provided)
            endpoint: API endpoint (overrides config)
            api_key: API key (overrides config)
            timeout: Request timeout (overrides config)
            batch_size: Batch size (overrides config)
            flush_interval: Flush interval (overrides config)
        """
        if config is None:
            config = get_config()
        
        self.config = config
        self.endpoint = (endpoint or config.endpoint).rstrip("/")
        self.api_key = api_key or config.api_key
        self.timeout = timeout or config.timeout
        self.batch_size = batch_size or config.batch_size
        self.flush_interval = flush_interval or config.flush_interval
        
        self._events_queue: List[Event] = []
        self._client = httpx.AsyncClient(
            timeout=self.timeout,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/x-protobuf",
                "User-Agent": "chaukas-sdk/0.1.0",
            }
        )
        self._flush_task: Optional[asyncio.Task] = None
        self._closed = False
    
    async def send_event(self, event: Union[Event, EventWrapper]) -> None:
        """
        Send a single event to the platform.
        
        Args:
            event: Proto Event or EventWrapper instance
        """
        if self._closed:
            logger.warning("Client is closed, ignoring event")
            return
        
        # Convert wrapper to proto if needed
        if isinstance(event, EventWrapper):
            proto_event = event.to_proto()
        else:
            proto_event = event
        
        self._events_queue.append(proto_event)
        
        if len(self._events_queue) >= self.batch_size:
            await self._flush_events()
    
    async def send_events(self, events: List[Union[Event, EventWrapper]]) -> None:
        """
        Send multiple events in batch.
        
        Args:
            events: List of proto Events or EventWrapper instances
        """
        if self._closed:
            logger.warning("Client is closed, ignoring events")
            return
        
        # Convert wrappers to proto if needed
        proto_events = []
        for event in events:
            if isinstance(event, EventWrapper):
                proto_events.append(event.to_proto())
            else:
                proto_events.append(event)
        
        self._events_queue.extend(proto_events)
        
        if len(self._events_queue) >= self.batch_size:
            await self._flush_events()
    
    async def _flush_events(self) -> None:
        """Flush queued events to the platform using proto messages."""
        if not self._events_queue:
            return
        
        events_to_send = self._events_queue[:]
        self._events_queue.clear()
        
        try:
            if len(events_to_send) == 1:
                # Send single event
                request = IngestEventRequest()
                request.event.CopyFrom(events_to_send[0])
                
                response = await self._client.post(
                    f"{self.endpoint}/v1/events/ingest",
                    content=request.SerializeToString()
                )
            else:
                # Send batch of events
                batch = EventBatch()
                batch.events.extend(events_to_send)
                
                # Set batch metadata
                from ..utils.uuid7 import generate_uuid7
                from google.protobuf import timestamp_pb2
                
                batch.batch_id = generate_uuid7()
                batch.timestamp.GetCurrentTime()
                
                request = IngestEventBatchRequest()
                request.event_batch.CopyFrom(batch)
                
                response = await self._client.post(
                    f"{self.endpoint}/v1/events/ingest-batch",
                    content=request.SerializeToString()
                )
            
            response.raise_for_status()
            
            logger.debug(f"Successfully sent {len(events_to_send)} events")
            
        except Exception as e:
            logger.error(f"Failed to send events: {e}")
            # Re-queue events for retry (simple strategy)
            self._events_queue = events_to_send + self._events_queue
    
    def create_event_builder(self) -> "EventBuilder":
        """
        Create an EventBuilder instance configured with this client.
        
        Returns:
            EventBuilder instance ready to create proto events
        """
        from .event_builder import EventBuilder
        return EventBuilder()
    
    def create_event_wrapper(self, event: Optional[Event] = None) -> EventWrapper:
        """
        Create an EventWrapper instance.
        
        Args:
            event: Optional proto Event to wrap
            
        Returns:
            EventWrapper instance
        """
        return EventWrapper(event)
    
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