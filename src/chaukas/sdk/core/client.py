"""
Chaukas client implementation using proto messages for 100% spec compliance.
"""

import asyncio
import json
import logging
from typing import Optional, List, Union
from datetime import datetime

import httpx
from google.protobuf.json_format import MessageToDict

from chaukas.spec.common.v1.events_pb2 import Event, EventBatch
from chaukas.spec.client.v1.client_pb2 import IngestEventRequest, IngestEventBatchRequest

from chaukas.sdk.core.config import ChaukasConfig, get_config
from chaukas.sdk.core.proto_wrapper import EventWrapper

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
        self._queue_size_bytes = 0  # Track approximate size of queued events
        self._client = httpx.AsyncClient(
            timeout=self.timeout,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
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
        
        # Estimate event size
        event_size = self._estimate_event_size(proto_event)
        
        self._events_queue.append(proto_event)
        self._queue_size_bytes += event_size
        
        # Flush if we hit count or size threshold
        should_flush = (
            len(self._events_queue) >= self.batch_size or
            (self.config.enable_adaptive_batching and 
             self._queue_size_bytes >= self.config.max_batch_bytes)
        )
        
        if should_flush:
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
        total_size = 0
        for event in events:
            if isinstance(event, EventWrapper):
                proto_event = event.to_proto()
            else:
                proto_event = event
            proto_events.append(proto_event)
            total_size += self._estimate_event_size(proto_event)
        
        self._events_queue.extend(proto_events)
        self._queue_size_bytes += total_size
        
        # Flush if we hit count or size threshold
        should_flush = (
            len(self._events_queue) >= self.batch_size or
            (self.config.enable_adaptive_batching and 
             self._queue_size_bytes >= self.config.max_batch_bytes)
        )
        
        if should_flush:
            await self._flush_events()
    
    async def _flush_events(self) -> None:
        """Flush queued events to the platform or file."""
        if not self._events_queue:
            return
        
        events_to_send = self._events_queue[:]
        self._events_queue.clear()
        self._queue_size_bytes = 0
        
        try:
            if self.config.output_mode == "file":
                await self._write_events_to_file(events_to_send)
            else:
                await self._send_events_to_api_with_retry(events_to_send)
                
            logger.debug(f"Successfully sent {len(events_to_send)} events")
            
        except Exception as e:
            logger.error(f"Failed to send events: {e}")
            # Re-queue events for retry with updated size
            self._events_queue = events_to_send + self._events_queue
            self._queue_size_bytes = sum(self._estimate_event_size(e) for e in self._events_queue)
    
    async def _send_events_to_api_with_retry(self, events: List[Event]) -> None:
        """Send events to API with retry and batch splitting on 503."""
        batch_size = len(events)
        attempt = 0
        max_attempts = 3
        
        while attempt < max_attempts:
            try:
                await self._send_events_to_api(events)
                return  # Success
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 503 and "high memory usage" in str(e.response.text).lower():
                    # High memory error - split batch and retry
                    attempt += 1
                    if batch_size > self.config.min_batch_size:
                        # Split batch in half
                        new_batch_size = max(batch_size // 2, self.config.min_batch_size)
                        logger.warning(
                            f"503 high memory error, retrying with smaller batch "
                            f"(was {batch_size}, now {new_batch_size})"
                        )
                        
                        # Send in smaller chunks
                        for i in range(0, len(events), new_batch_size):
                            chunk = events[i:i + new_batch_size]
                            await self._send_events_to_api(chunk)
                        return  # All chunks sent successfully
                    else:
                        logger.error(f"Cannot reduce batch size further (already at {batch_size})")
                        raise
                else:
                    # Other HTTP error - don't retry
                    raise
            except Exception as e:
                # Non-HTTP error - don't retry
                logger.error(f"Failed to send batch: {e}")
                raise
    
    async def _send_events_to_api(self, events: List[Event]) -> None:
        """Send events to API endpoint using JSON format compatible with chaukas-core."""
        if len(events) == 1:
            # Send single event to /events endpoint
            request = IngestEventRequest()
            request.event.CopyFrom(events[0])

            # Convert protobuf to JSON dict
            request_dict = MessageToDict(request, preserving_proto_field_name=True)

            response = await self._client.post(
                f"{self.endpoint}/events",
                json=request_dict
            )
        else:
            # Send batch of events to /events/batch endpoint
            batch = EventBatch()
            batch.events.extend(events)

            # Set batch metadata
            from chaukas.sdk.utils.uuid7 import generate_uuid7

            batch.batch_id = generate_uuid7()
            batch.timestamp.GetCurrentTime()

            request = IngestEventBatchRequest()
            request.event_batch.CopyFrom(batch)

            # Convert protobuf to JSON dict
            request_dict = MessageToDict(request, preserving_proto_field_name=True)

            response = await self._client.post(
                f"{self.endpoint}/events/batch",
                json=request_dict
            )

        response.raise_for_status()
    
    async def _write_events_to_file(self, events: List[Event]) -> None:
        """Write events to file in JSON Lines format."""
        import aiofiles
        
        async with aiofiles.open(self.config.output_file, "a") as f:
            for event in events:
                # Convert proto to dict for JSON serialization
                event_dict = MessageToDict(event, preserving_proto_field_name=True)
                event_dict["timestamp"] = datetime.utcnow().isoformat()
                
                # Write as JSON line
                await f.write(json.dumps(event_dict) + "\n")
    
    def create_event_builder(self) -> "EventBuilder":
        """
        Create an EventBuilder instance configured with this client.
        
        Returns:
            EventBuilder instance ready to create proto events
        """
        from chaukas.sdk.core.event_builder import EventBuilder
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
    
    def _estimate_event_size(self, event: Event) -> int:
        """
        Estimate the size of an event in bytes.
        
        This is an approximation based on field sizes to avoid the overhead
        of actual serialization for every event.
        
        Args:
            event: Proto Event to estimate
            
        Returns:
            Estimated size in bytes
        """
        # Base size for required fields
        size = 200  # Base overhead for proto structure
        
        # String fields
        size += len(event.event_id) + len(event.tenant_id) + len(event.project_id)
        size += len(event.session_id) + len(event.trace_id) + len(event.span_id)
        
        if event.parent_span_id:
            size += len(event.parent_span_id)
        if event.agent_id:
            size += len(event.agent_id)
        if event.agent_name:
            size += len(event.agent_name)
        if event.branch:
            size += len(event.branch)
        
        # Tags
        for tag in event.tags:
            size += len(tag)
        
        # Metadata (estimate based on proto structure)
        if event.HasField('metadata'):
            # Rough estimate for struct fields
            size += 500
        
        # Content fields - estimate based on type
        if event.HasField('message'):
            # Message content
            size += 200
            if hasattr(event.message, 'text'):
                size += len(event.message.text)
        if event.HasField('llm_invocation'):
            # LLM invocations can be large due to messages
            size += 1000  # Base size
            # Add size for request/response structs if they exist
            size += 500  # Rough estimate
        if event.HasField('tool_call'):
            size += 300  # Tool calls are typically smaller
        if event.HasField('tool_response'):
            size += 500  # Tool responses can vary
        if event.HasField('mcp_call'):
            size += 400  # MCP calls
        if event.HasField('agent_handoff'):
            size += 300  # Agent handoffs
        if event.HasField('error'):
            size += 200  # Error information
        if event.HasField('retry'):
            size += 150  # Retry information
        
        return size