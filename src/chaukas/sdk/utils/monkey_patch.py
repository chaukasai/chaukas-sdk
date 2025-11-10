"""
Monkey patching framework for instrumenting agent SDKs.
"""

import importlib
import logging
import sys
from functools import wraps
from typing import Any, Callable, Dict, List, Optional

import wrapt

from chaukas.sdk.core.tracer import ChaukasTracer

logger = logging.getLogger(__name__)


class PatchTarget:
    """Represents a target for monkey patching."""

    def __init__(
        self,
        module_name: str,
        class_name: Optional[str],
        method_name: str,
        wrapper_func: Callable,
        condition: Optional[Callable[[], bool]] = None,
    ):
        self.module_name = module_name
        self.class_name = class_name
        self.method_name = method_name
        self.wrapper_func = wrapper_func
        self.condition = condition
        self.original_func: Optional[Callable] = None
        self.is_patched = False


class MonkeyPatcher:
    """Manages monkey patching for multiple agent SDKs."""

    def __init__(self, tracer: ChaukasTracer, config: Dict[str, Any]):
        self.tracer = tracer
        self.config = config
        self.patches: List[PatchTarget] = []
        self._wrappers: List[Any] = []  # Store wrapper instances for cleanup
        self._auto_detect = config.get("auto_detect", True)
        self._enabled_integrations = config.get("enabled_integrations", [])

    def patch_all(self) -> None:
        """Apply all relevant monkey patches based on detected SDKs."""
        if self._auto_detect:
            self._detect_and_patch_sdks()
        else:
            self._patch_enabled_integrations()

    def unpatch_all(self) -> None:
        """Remove all applied monkey patches."""
        for patch in self.patches:
            if patch.is_patched:
                self._unpatch_target(patch)

    async def close(self) -> None:
        """Close all wrappers and clean up resources."""
        import asyncio

        for wrapper in self._wrappers:
            if hasattr(wrapper, "close"):
                if asyncio.iscoroutinefunction(wrapper.close):
                    await wrapper.close()
                else:
                    wrapper.close()
        self._wrappers.clear()

    def _detect_and_patch_sdks(self) -> None:
        """Auto-detect installed SDKs and patch them."""
        sdk_modules = {
            "openai_agents": "agents",
            "google_adk": "adk",
            "crewai": "crewai",
        }

        for sdk_name, module_name in sdk_modules.items():
            try:
                importlib.import_module(module_name)
                logger.info(f"Detected {sdk_name}, applying patches")
                getattr(self, f"_patch_{sdk_name}")()
            except ImportError:
                logger.debug(f"{sdk_name} not found, skipping")

    def _patch_enabled_integrations(self) -> None:
        """Patch only explicitly enabled integrations."""
        for integration in self._enabled_integrations:
            patch_method = f"_patch_{integration}"
            if hasattr(self, patch_method):
                getattr(self, patch_method)()

    def _patch_google_adk(self) -> None:
        """Apply patches for Google ADK."""
        from chaukas.sdk.integrations.google_adk import GoogleADKWrapper

        wrapper = GoogleADKWrapper(self.tracer)

        # Patch Agent execution methods
        self._add_patch(
            "adk",
            "Agent",
            "run",
            wrapper.wrap_agent_run,
        )

        self._add_patch(
            "adk",
            "LlmAgent",
            "run",
            wrapper.wrap_llm_agent_run,
        )

    def _patch_openai_agents(self) -> None:
        """Apply patches for OpenAI Agents."""
        from chaukas.sdk.integrations.openai_agents import OpenAIAgentsWrapper

        wrapper = OpenAIAgentsWrapper(self.tracer)
        self._wrappers.append(wrapper)  # Store wrapper for cleanup

        # Apply patches to Runner methods
        wrapper.patch_runner()

        # Apply patches to MCP Server methods (if available)
        wrapper.patch_mcp_server()

    def _patch_crewai(self) -> None:
        """Apply patches for CrewAI."""
        from chaukas.sdk.integrations.crewai import CrewAIWrapper

        wrapper = CrewAIWrapper(self.tracer)

        # Apply direct patches
        wrapper.patch_crew()
        wrapper.patch_agent()

    def _add_patch(
        self,
        module_name: str,
        class_name: Optional[str],
        method_name: str,
        wrapper_func: Callable,
        condition: Optional[Callable[[], bool]] = None,
    ) -> None:
        """Add a patch target and apply it if possible."""
        patch = PatchTarget(
            module_name=module_name,
            class_name=class_name,
            method_name=method_name,
            wrapper_func=wrapper_func,
            condition=condition,
        )

        self.patches.append(patch)
        self._apply_patch(patch)

    def _apply_patch(self, patch: PatchTarget) -> None:
        """Apply a single monkey patch."""
        try:
            # Check condition if provided
            if patch.condition and not patch.condition():
                logger.debug(
                    f"Condition not met for {patch.module_name}.{patch.class_name}.{patch.method_name}"
                )
                return

            module = importlib.import_module(patch.module_name)

            if patch.class_name:
                target_class = getattr(module, patch.class_name)
                target_method = getattr(target_class, patch.method_name)

                # Store original for restoration
                patch.original_func = target_method

                # Apply wrapt patch
                wrapt.wrap_function_wrapper(
                    module,
                    f"{patch.class_name}.{patch.method_name}",
                    patch.wrapper_func,
                )
            else:
                target_func = getattr(module, patch.method_name)
                patch.original_func = target_func

                wrapt.wrap_function_wrapper(
                    module, patch.method_name, patch.wrapper_func
                )

            patch.is_patched = True
            logger.debug(
                f"Patched {patch.module_name}.{patch.class_name}.{patch.method_name}"
            )

        except Exception as e:
            logger.error(
                f"Failed to patch {patch.module_name}.{patch.class_name}.{patch.method_name}: {e}"
            )

    def _unpatch_target(self, patch: PatchTarget) -> None:
        """Remove a single monkey patch."""
        try:
            if not patch.original_func:
                return

            module = importlib.import_module(patch.module_name)

            if patch.class_name:
                target_class = getattr(module, patch.class_name)
                setattr(target_class, patch.method_name, patch.original_func)
            else:
                setattr(module, patch.method_name, patch.original_func)

            patch.is_patched = False
            logger.debug(
                f"Unpatched {patch.module_name}.{patch.class_name}.{patch.method_name}"
            )

        except Exception as e:
            logger.error(
                f"Failed to unpatch {patch.module_name}.{patch.class_name}.{patch.method_name}: {e}"
            )


def create_wrapper(event_type: str, source: str, normalize_for: Optional[str] = None):
    """Decorator factory for creating instrumentation wrappers."""

    def decorator(func):
        @wraps(func)
        async def async_wrapper(wrapped, instance, args, kwargs):
            from chaukas.sdk import get_tracer

            tracer = get_tracer()
            if not tracer:
                return await wrapped(*args, **kwargs)

            with tracer.start_span(f"{source}.{func.__name__}") as span:
                span.set_attribute("function", func.__name__)
                span.set_attribute("source", source)

                try:
                    # Send start event
                    start_data = {
                        "function": func.__name__,
                        "args": _serialize_args(args),
                        "kwargs": _serialize_kwargs(kwargs),
                    }

                    await tracer.send_event(
                        event_type=f"{event_type}.start",
                        source=source,
                        data=start_data,
                        normalize_for=normalize_for,
                    )

                    result = await wrapped(*args, **kwargs)

                    # Send success event
                    end_data = {
                        "function": func.__name__,
                        "result": _serialize_result(result),
                    }

                    await tracer.send_event(
                        event_type=f"{event_type}.end",
                        source=source,
                        data=end_data,
                        normalize_for=normalize_for,
                    )

                    return result

                except Exception as e:
                    # Send error event
                    error_data = {
                        "function": func.__name__,
                        "error": str(e),
                        "error_type": type(e).__name__,
                    }

                    await tracer.send_event(
                        event_type=f"{event_type}.error",
                        source=source,
                        data=error_data,
                        normalize_for=normalize_for,
                    )

                    raise

        @wraps(func)
        def sync_wrapper(wrapped, instance, args, kwargs):
            from chaukas.sdk import get_tracer

            tracer = get_tracer()
            if not tracer:
                return wrapped(*args, **kwargs)

            with tracer.start_span(f"{source}.{func.__name__}") as span:
                span.set_attribute("function", func.__name__)
                span.set_attribute("source", source)

                try:
                    result = wrapped(*args, **kwargs)
                    return result

                except Exception as e:
                    span.set_status("error", str(e))
                    raise

        # Return appropriate wrapper based on function type
        if hasattr(func, "__code__") and func.__code__.co_flags & 0x80:  # CO_COROUTINE
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


def _serialize_args(args) -> List[Any]:
    """Serialize function arguments for logging."""
    try:
        return [_serialize_value(arg) for arg in args]
    except Exception:
        return ["<serialization_error>"]


def _serialize_kwargs(kwargs) -> Dict[str, Any]:
    """Serialize function keyword arguments for logging."""
    try:
        return {k: _serialize_value(v) for k, v in kwargs.items()}
    except Exception:
        return {"<serialization_error>": True}


def _serialize_result(result) -> Any:
    """Serialize function result for logging."""
    try:
        return _serialize_value(result)
    except Exception:
        return "<serialization_error>"


def _serialize_value(value) -> Any:
    """Serialize a single value, handling complex objects."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    elif isinstance(value, (list, tuple)):
        return [
            _serialize_value(item) for item in value[:10]
        ]  # Limit to first 10 items
    elif isinstance(value, dict):
        return {
            k: _serialize_value(v) for k, v in list(value.items())[:10]
        }  # Limit to first 10 items
    else:
        return f"<{type(value).__name__}>"
