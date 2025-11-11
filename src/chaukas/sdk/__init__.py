"""
Chaukas SDK - One-line instrumentation for agent building SDKs.
"""

import asyncio
import atexit
import logging
import os
import weakref
from typing import Any, Dict, Optional

# Load .env file automatically
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    # dotenv not installed, skip loading
    pass

from chaukas.sdk.core.agent_mapper import AgentMapper
from chaukas.sdk.core.client import ChaukasClient
from chaukas.sdk.core.config import ChaukasConfig, get_config, set_config
from chaukas.sdk.core.event_builder import EventBuilder
from chaukas.sdk.core.proto_wrapper import EventWrapper
from chaukas.sdk.core.tracer import ChaukasTracer
from chaukas.sdk.utils.monkey_patch import MonkeyPatcher

__version__ = "0.1.0"

# Export proto messages for advanced usage
try:
    from chaukas.spec.client.v1 import client_pb2
    from chaukas.spec.common.v1 import events_pb2
except ImportError as e:
    import logging

    logging.warning(
        f"Failed to import proto modules: {e}. Proto features will not be available."
    )
    events_pb2 = None
    client_pb2 = None

__all__ = [
    "enable_chaukas",
    "disable_chaukas",
    "is_enabled",
    "get_tracer",
    "get_client",
    "get_langchain_callback",
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
    auto_flush: bool = True,
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
        auto_flush: Automatically flush events (default: True)
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

        # Apply monkey patches with auto-flush enabled
        _patcher.patch_all()

        _enabled = True

        # Register cleanup handlers
        if auto_flush:
            atexit.register(_cleanup_on_exit)
            # Patch asyncio.run() to auto-flush before event loop closes
            _patch_asyncio_run()

        logger.info("Chaukas instrumentation enabled with proto compliance")

    except Exception as e:
        logger.error(f"Failed to enable Chaukas: {e}")
        raise


# Store original asyncio.run
_original_asyncio_run = asyncio.run


def _patched_asyncio_run(main, *, debug=None):
    """Patched version of asyncio.run that flushes events before closing loop."""
    global _client

    async def _wrapper():
        try:
            result = await main
            # Flush events before loop closes
            if _client and _enabled:
                try:
                    await _client.flush()
                except Exception as e:
                    logger.debug(f"Failed to flush events: {e}")
            return result
        except Exception:
            # Still try to flush on error
            if _client and _enabled:
                try:
                    await _client.flush()
                except Exception as e:
                    logger.debug(f"Failed to flush events on error: {e}")
            raise

    return _original_asyncio_run(_wrapper(), debug=debug)


def _patch_asyncio_run() -> None:
    """Patch asyncio.run to auto-flush events."""
    asyncio.run = _patched_asyncio_run


def _cleanup_on_exit() -> None:
    """Cleanup handler called on program exit to flush events."""
    global _enabled

    if _enabled:
        logger.debug("Flushing events on program exit...")
        _close_client_sync()


def _close_client_sync() -> None:
    """Helper to close client from sync context."""
    global _client

    if _client is None:
        return

    try:
        # Check if there's an existing event loop
        try:
            loop = asyncio.get_running_loop()
            # We're in an async context, create a task
            asyncio.create_task(_client.close())
        except RuntimeError:
            # No running loop - we need to create a new one
            # However, during interpreter shutdown, we can't create new loops
            # So we'll try our best with a new loop, but catch all errors
            try:
                # Create a new event loop specifically for cleanup
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    loop.run_until_complete(_client.close())
                finally:
                    loop.close()
            except Exception as e:
                # During interpreter shutdown, even this may fail
                # Log but don't raise - we're shutting down anyway
                logger.debug(f"Could not properly close client during shutdown: {e}")
    except Exception as e:
        logger.debug(f"Cleanup failed: {e}")


def disable_chaukas() -> None:
    """Disable Chaukas instrumentation and restore original methods."""
    global _client, _tracer, _patcher, _enabled

    if not _enabled:
        return

    # Remove LangChain auto-instrumentation if present
    if _patcher:
        for wrapper in _patcher._wrappers:
            if hasattr(wrapper, "remove_auto_instrument"):
                try:
                    wrapper.remove_auto_instrument()
                except Exception as e:
                    logger.debug(f"Failed to remove auto-instrumentation: {e}")

    # Close client to flush events
    _close_client_sync()

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


def get_langchain_callback():
    """
    Get the LangChain callback handler for instrumentation.

    This handler should be passed to LangChain chains/agents via the callbacks parameter.

    Returns:
        BaseCallbackHandler instance configured for Chaukas event capture.

    Example:
        >>> import chaukas
        >>> chaukas.enable_chaukas()
        >>> callback = chaukas.get_langchain_callback()
        >>>
        >>> # Use with chains
        >>> chain.invoke(input, config={'callbacks': [callback]})
        >>>
        >>> # Use with agents
        >>> agent_executor.invoke(input, config={'callbacks': [callback]})

    Raises:
        RuntimeError: If Chaukas is not enabled. Call enable_chaukas() first.
    """
    global _patcher

    if not _enabled:
        raise RuntimeError(
            "Chaukas is not enabled. Call chaukas.enable_chaukas() first."
        )

    if _patcher is None:
        raise RuntimeError("MonkeyPatcher not initialized.")

    # Find the LangChain wrapper in the patcher's wrappers
    for wrapper in _patcher._wrappers:
        if hasattr(wrapper, "get_callback_handler"):
            return wrapper.get_callback_handler()

    # If not found, try to create it
    try:
        from chaukas.sdk.integrations.langchain import LangChainWrapper

        wrapper = LangChainWrapper(_tracer)
        _patcher._wrappers.append(wrapper)
        return wrapper.get_callback_handler()
    except ImportError:
        raise RuntimeError(
            "LangChain integration not available. Is LangChain installed?"
        )
