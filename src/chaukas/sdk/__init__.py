"""
Chaukas SDK - One-line instrumentation for agent building SDKs.
"""

import os
import logging
from typing import Optional, Dict, Any

from chaukas.sdk.core.client import ChaukasClient
from chaukas.sdk.core.tracer import ChaukasTracer
from chaukas.sdk.core.config import ChaukasConfig, get_config, set_config
from chaukas.sdk.core.event_builder import EventBuilder
from chaukas.sdk.core.proto_wrapper import EventWrapper
from chaukas.sdk.core.agent_mapper import AgentMapper
from chaukas.sdk.utils.monkey_patch import MonkeyPatcher

__version__ = "0.1.0"

# Export proto messages for advanced usage
try:
    from chaukas.spec.common.v1 import events_pb2
    from chaukas.spec.client.v1 import client_pb2
except ImportError as e:
    import logging
    logging.warning(f"Failed to import proto modules: {e}. Proto features will not be available.")
    events_pb2 = None
    client_pb2 = None

__all__ = [
    "enable_chaukas",
    "disable_chaukas", 
    "is_enabled",
    "get_tracer",
    "get_client",
    "ChaukasClient",
    "ChaukasTracer",
    "ChaukasConfig",
    "EventBuilder",
    "EventWrapper",
    "AgentMapper",
    "events_pb2",
    "client_pb2",
]

logger = logging.getLogger(__name__)

_client: Optional[ChaukasClient] = None
_tracer: Optional[ChaukasTracer] = None
_patcher: Optional[MonkeyPatcher] = None
_enabled = False


def enable_chaukas(
    endpoint: Optional[str] = None,
    api_key: Optional[str] = None,
    tenant_id: Optional[str] = None,
    project_id: Optional[str] = None,
    session_id: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Enable Chaukas instrumentation for agent SDKs with proto compliance.
    
    Required environment variables (if not provided as parameters):
    - CHAUKAS_ENDPOINT: API endpoint
    - CHAUKAS_API_KEY: API key  
    - CHAUKAS_TENANT_ID: Tenant ID
    - CHAUKAS_PROJECT_ID: Project ID
    
    Args:
        endpoint: Override CHAUKAS_ENDPOINT environment variable
        api_key: Override CHAUKAS_API_KEY environment variable
        tenant_id: Override CHAUKAS_TENANT_ID environment variable
        project_id: Override CHAUKAS_PROJECT_ID environment variable
        session_id: Optional session ID for tracing
        config: Additional configuration options
    """
    global _client, _tracer, _patcher, _enabled
    
    if _enabled:
        logger.warning("Chaukas is already enabled")
        return
    
    try:
        # Create or override configuration
        if any([endpoint, api_key, tenant_id, project_id]):
            # Create config from parameters and environment
            chaukas_config = ChaukasConfig(
                endpoint=endpoint or os.getenv("CHAUKAS_ENDPOINT"),
                api_key=api_key or os.getenv("CHAUKAS_API_KEY"),
                tenant_id=tenant_id or os.getenv("CHAUKAS_TENANT_ID"),
                project_id=project_id or os.getenv("CHAUKAS_PROJECT_ID"),
                # Use defaults for other fields
            )
            set_config(chaukas_config)
        else:
            # Use environment configuration
            chaukas_config = get_config()
        
        # Initialize core components
        _client = ChaukasClient(config=chaukas_config)
        _tracer = ChaukasTracer(client=_client, session_id=session_id)
        _patcher = MonkeyPatcher(tracer=_tracer, config=config or {})
        
        # Apply monkey patches
        _patcher.patch_all()
        
        _enabled = True
        logger.info("Chaukas instrumentation enabled with proto compliance")
        
    except Exception as e:
        logger.error(f"Failed to enable Chaukas: {e}")
        raise


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