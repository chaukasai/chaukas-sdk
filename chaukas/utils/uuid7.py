"""
UUID7 generation utility for time-ordered unique identifiers.
"""

import time
import random
import uuid
from typing import Optional

try:
    from uuid6 import uuid7
    HAS_UUID6 = True
except ImportError:
    HAS_UUID6 = False


def generate_uuid7() -> str:
    """
    Generate a UUID7 (time-ordered UUID) string.
    
    UUID7 provides time-ordering while maintaining uniqueness,
    which is ideal for event IDs in distributed systems.
    
    Returns:
        str: A UUID7 string
    """
    if HAS_UUID6:
        # Use the uuid6 package if available
        return str(uuid7())
    else:
        # Fallback implementation of UUID7-like functionality
        # This creates a time-based UUID that approximates UUID7 behavior
        return _generate_uuid7_fallback()


def _generate_uuid7_fallback() -> str:
    """
    Fallback implementation for UUID7-like generation.
    
    Creates a time-ordered UUID by encoding timestamp in the first part
    and random bits in the second part.
    """
    # Get current timestamp in milliseconds
    timestamp_ms = int(time.time() * 1000)
    
    # Create 128-bit value
    # First 48 bits: timestamp (milliseconds since epoch)
    # Next 12 bits: version (0111 for v7) and random
    # Next 2 bits: variant (10)
    # Remaining 62 bits: random
    
    # Timestamp (48 bits)
    time_hi = (timestamp_ms >> 16) & 0xFFFFFFFF
    time_lo = timestamp_ms & 0xFFFF
    
    # Version 7 and random bits
    clock_seq_hi_variant = (0x7 << 4) | (random.getrandbits(4))
    clock_seq_low = random.getrandbits(8)
    
    # Node (48 bits of randomness)
    node = random.getrandbits(48)
    
    # Construct UUID fields
    fields = (
        time_hi,
        time_lo,
        (clock_seq_hi_variant << 8) | clock_seq_low,
        node >> 32,
        node & 0xFFFFFFFF
    )
    
    # Format as standard UUID string
    return "%08x-%04x-%04x-%04x-%012x" % fields


def generate_uuid7_from_timestamp(timestamp_ms: int) -> str:
    """
    Generate a UUID7 from a specific timestamp.
    
    Useful for creating deterministic UUIDs for testing or
    when you need to preserve the original event time.
    
    Args:
        timestamp_ms: Timestamp in milliseconds since epoch
        
    Returns:
        str: A UUID7 string based on the provided timestamp
    """
    if HAS_UUID6:
        # uuid6 package doesn't directly support custom timestamps
        # Fall back to our implementation
        pass
    
    # Use our implementation with specific timestamp
    time_hi = (timestamp_ms >> 16) & 0xFFFFFFFF
    time_lo = timestamp_ms & 0xFFFF
    
    clock_seq_hi_variant = (0x7 << 4) | (random.getrandbits(4))
    clock_seq_low = random.getrandbits(8)
    
    node = random.getrandbits(48)
    
    fields = (
        time_hi,
        time_lo,
        (clock_seq_hi_variant << 8) | clock_seq_low,
        node >> 32,
        node & 0xFFFFFFFF
    )
    
    return "%08x-%04x-%04x-%04x-%012x" % fields