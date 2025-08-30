"""
Chaukas SDK - One-line instrumentation for agent building SDKs.
"""

import os
import logging
from typing import Optional, Dict, Any

from .core.client import ChaukasClient
from .core.tracer import ChaukasTracer
from .utils.monkey_patch import MonkeyPatcher

__version__ = "0.1.0"

logger = logging.getLogger(__name__)

_client: Optional[ChaukasClient] = None
_tracer: Optional[ChaukasTracer] = None
_patcher: Optional[MonkeyPatcher] = None
_enabled = False


def enable_chaukas(
    endpoint: Optional[str] = None,
    api_key: Optional[str] = None,
    session_id: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Enable Chaukas instrumentation for agent SDKs.
    
    Args:
        endpoint: Override CHAUKAS_ENDPOINT environment variable
        api_key: Override CHAUKAS_API_KEY environment variable  
        session_id: Optional session ID for tracing
        config: Additional configuration options
    """
    global _client, _tracer, _patcher, _enabled
    
    if _enabled:
        logger.warning("Chaukas is already enabled")
        return
    
    # Get configuration from environment or parameters
    endpoint = endpoint or os.getenv("CHAUKAS_ENDPOINT")
    api_key = api_key or os.getenv("CHAUKAS_API_KEY")
    
    if not endpoint:
        raise ValueError("CHAUKAS_ENDPOINT environment variable or endpoint parameter is required")
    if not api_key:
        raise ValueError("CHAUKAS_API_KEY environment variable or api_key parameter is required")
    
    # Initialize core components
    _client = ChaukasClient(endpoint=endpoint, api_key=api_key)
    _tracer = ChaukasTracer(client=_client, session_id=session_id)
    _patcher = MonkeyPatcher(tracer=_tracer, config=config or {})
    
    # Apply monkey patches
    _patcher.patch_all()
    
    _enabled = True
    logger.info("Chaukas instrumentation enabled")


def disable_chaukas() -> None:
    """Disable Chaukas instrumentation and restore original methods."""
    global _client, _tracer, _patcher, _enabled
    
    if not _enabled:
        return
    
    if _patcher:
        _patcher.unpatch_all()
    
    _client = None
    _tracer = None
    _patcher = None
    _enabled = False
    
    logger.info("Chaukas instrumentation disabled")


def is_enabled() -> bool:
    """Check if Chaukas instrumentation is currently enabled."""
    return _enabled


def get_tracer() -> Optional[ChaukasTracer]:
    """Get the current tracer instance."""
    return _tracer


def get_client() -> Optional[ChaukasClient]:
    """Get the current client instance."""
    return _client