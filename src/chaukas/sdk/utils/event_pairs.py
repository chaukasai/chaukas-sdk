"""
Event pair management for START/END event correlation.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class EventPairInfo:
    """Information about a START/END event pair."""

    span_id: str
    event_type: str
    start_time: datetime
    metadata: Optional[Dict] = None


class EventPairManager:
    """
    Manages START/END event pairs with proper span ID correlation.

    This ensures that related START and END events share the same span_id
    for proper distributed tracing and event correlation.
    """

    def __init__(self):
        """Initialize event pair manager."""
        # Maps (event_type, identifier) -> EventPairInfo
        self._active_pairs: Dict[Tuple[str, str], EventPairInfo] = {}

        # Track orphaned END events (END without START)
        self._orphaned_ends: Dict[Tuple[str, str], datetime] = {}

    def register_start_event(
        self,
        event_type: str,
        identifier: str,
        span_id: str,
        metadata: Optional[Dict] = None,
    ) -> str:
        """
        Register a START event and store its span_id for the corresponding END event.

        Args:
            event_type: Type of event (e.g., "AGENT", "SESSION", "MODEL_INVOCATION")
            identifier: Unique identifier for this event pair (e.g., agent_id, session_id)
            span_id: Span ID assigned to the START event
            metadata: Optional metadata to store with the pair

        Returns:
            The span_id (for convenience)
        """
        key = (event_type, identifier)

        # Check if there's already an active pair (potential issue)
        if key in self._active_pairs:
            logger.warning(
                f"START event registered for {event_type}:{identifier} but pair already exists. "
                f"Overwriting previous pair."
            )

        # Store pair info
        self._active_pairs[key] = EventPairInfo(
            span_id=span_id,
            event_type=event_type,
            start_time=datetime.now(timezone.utc),
            metadata=metadata,
        )

        logger.debug(
            f"Registered START event for {event_type}:{identifier} with span_id {span_id}"
        )

        return span_id

    def get_span_id_for_end(
        self, event_type: str, identifier: str, clear: bool = True
    ) -> Optional[str]:
        """
        Get the span_id for an END event based on its corresponding START event.

        Args:
            event_type: Type of event
            identifier: Unique identifier for this event pair
            clear: Whether to clear the pair after retrieval (default: True)

        Returns:
            Span ID if START event was registered, None otherwise
        """
        key = (event_type, identifier)

        pair_info = self._active_pairs.get(key)

        if pair_info:
            span_id = pair_info.span_id

            if clear:
                # Remove the pair since END event is being sent
                del self._active_pairs[key]
                logger.debug(f"Cleared pair for {event_type}:{identifier}")

            return span_id
        else:
            # Track orphaned END event
            self._orphaned_ends[key] = datetime.now(timezone.utc)
            logger.warning(
                f"END event for {event_type}:{identifier} has no corresponding START event"
            )
            return None

    def clear_pair(self, event_type: str, identifier: str) -> bool:
        """
        Clear a START/END pair (e.g., on error).

        Args:
            event_type: Type of event
            identifier: Unique identifier for this event pair

        Returns:
            True if pair was cleared, False if it didn't exist
        """
        key = (event_type, identifier)

        if key in self._active_pairs:
            del self._active_pairs[key]
            logger.debug(f"Manually cleared pair for {event_type}:{identifier}")
            return True

        return False

    def get_active_pairs(self) -> Dict[Tuple[str, str], EventPairInfo]:
        """
        Get all currently active START/END pairs.

        Returns:
            Dictionary of active pairs
        """
        return dict(self._active_pairs)

    def get_orphaned_ends(self) -> Dict[Tuple[str, str], datetime]:
        """
        Get all orphaned END events (END without START).

        Returns:
            Dictionary of orphaned END events with timestamps
        """
        return dict(self._orphaned_ends)

    def cleanup_stale_pairs(self, max_age_seconds: int = 3600) -> int:
        """
        Clean up stale START events that never received an END event.

        Args:
            max_age_seconds: Maximum age in seconds before considering a pair stale

        Returns:
            Number of stale pairs cleaned up
        """
        current_time = datetime.now(timezone.utc)
        stale_pairs = []

        for key, pair_info in self._active_pairs.items():
            age = (current_time - pair_info.start_time).total_seconds()
            if age > max_age_seconds:
                stale_pairs.append(key)

        for key in stale_pairs:
            event_type, identifier = key
            logger.warning(
                f"Cleaning up stale pair for {event_type}:{identifier} "
                f"(no END event received after {max_age_seconds} seconds)"
            )
            del self._active_pairs[key]

        return len(stale_pairs)

    def cleanup_old_orphans(self, max_age_seconds: int = 3600) -> int:
        """
        Clean up old orphaned END event records.

        Args:
            max_age_seconds: Maximum age in seconds before removing orphan record

        Returns:
            Number of orphan records cleaned up
        """
        current_time = datetime.now(timezone.utc)
        old_orphans = []

        for key, timestamp in self._orphaned_ends.items():
            age = (current_time - timestamp).total_seconds()
            if age > max_age_seconds:
                old_orphans.append(key)

        for key in old_orphans:
            del self._orphaned_ends[key]

        return len(old_orphans)

    def get_stats(self) -> Dict[str, int]:
        """
        Get statistics about event pairs.

        Returns:
            Dictionary with stats
        """
        return {
            "active_pairs": len(self._active_pairs),
            "orphaned_ends": len(self._orphaned_ends),
            "total_tracked": len(self._active_pairs) + len(self._orphaned_ends),
        }

    def reset(self) -> None:
        """Clear all tracked pairs and orphans."""
        self._active_pairs.clear()
        self._orphaned_ends.clear()
        logger.debug("Event pair manager reset")
