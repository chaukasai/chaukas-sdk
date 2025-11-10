"""
Context utilities for managing trace context across async operations.
"""

import uuid
from contextvars import ContextVar, copy_context
from functools import wraps
from typing import Any, Dict, Optional

# Context variables for distributed tracing
session_id_var: ContextVar[Optional[str]] = ContextVar(
    "chaukas_session_id", default=None
)
trace_id_var: ContextVar[Optional[str]] = ContextVar("chaukas_trace_id", default=None)
span_id_var: ContextVar[Optional[str]] = ContextVar("chaukas_span_id", default=None)
parent_span_id_var: ContextVar[Optional[str]] = ContextVar(
    "chaukas_parent_span_id", default=None
)


def get_trace_context() -> Dict[str, Optional[str]]:
    """Get the current trace context."""
    return {
        "session_id": session_id_var.get(),
        "trace_id": trace_id_var.get(),
        "span_id": span_id_var.get(),
        "parent_span_id": parent_span_id_var.get(),
    }


def set_trace_context(
    session_id: Optional[str] = None,
    trace_id: Optional[str] = None,
    span_id: Optional[str] = None,
    parent_span_id: Optional[str] = None,
) -> None:
    """Set the trace context."""
    if session_id is not None:
        session_id_var.set(session_id)
    if trace_id is not None:
        trace_id_var.set(trace_id)
    if span_id is not None:
        span_id_var.set(span_id)
    if parent_span_id is not None:
        parent_span_id_var.set(parent_span_id)


def ensure_trace_context() -> Dict[str, str]:
    """Ensure trace context exists, generating IDs if necessary."""
    context = get_trace_context()

    if not context["session_id"]:
        context["session_id"] = str(uuid.uuid4())
        session_id_var.set(context["session_id"])

    if not context["trace_id"]:
        context["trace_id"] = str(uuid.uuid4())
        trace_id_var.set(context["trace_id"])

    if not context["span_id"]:
        context["span_id"] = str(uuid.uuid4())
        span_id_var.set(context["span_id"])

    return {k: v for k, v in context.items() if v is not None}


def with_trace_context(
    session_id: Optional[str] = None,
    trace_id: Optional[str] = None,
    span_id: Optional[str] = None,
    parent_span_id: Optional[str] = None,
):
    """Decorator to run function with specific trace context."""

    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Create new context with provided IDs
            ctx = copy_context()

            def run_with_context():
                set_trace_context(session_id, trace_id, span_id, parent_span_id)
                return func(*args, **kwargs)

            return await ctx.run(run_with_context)

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            ctx = copy_context()

            def run_with_context():
                set_trace_context(session_id, trace_id, span_id, parent_span_id)
                return func(*args, **kwargs)

            return ctx.run(run_with_context)

        # Return appropriate wrapper
        if hasattr(func, "__code__") and func.__code__.co_flags & 0x80:
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


class TraceContextManager:
    """Context manager for trace context."""

    def __init__(
        self,
        session_id: Optional[str] = None,
        trace_id: Optional[str] = None,
        span_id: Optional[str] = None,
        parent_span_id: Optional[str] = None,
    ):
        self.session_id = session_id
        self.trace_id = trace_id
        self.span_id = span_id
        self.parent_span_id = parent_span_id
        self._tokens = {}

    def __enter__(self):
        # Save current context and set new values
        current = get_trace_context()

        if self.session_id:
            self._tokens["session_id"] = session_id_var.set(self.session_id)
        if self.trace_id:
            self._tokens["trace_id"] = trace_id_var.set(self.trace_id)
        if self.span_id:
            self._tokens["span_id"] = span_id_var.set(self.span_id)
        if self.parent_span_id:
            self._tokens["parent_span_id"] = parent_span_id_var.set(self.parent_span_id)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore previous context
        for var_name, token in self._tokens.items():
            if var_name == "session_id":
                session_id_var.reset(token)
            elif var_name == "trace_id":
                trace_id_var.reset(token)
            elif var_name == "span_id":
                span_id_var.reset(token)
            elif var_name == "parent_span_id":
                parent_span_id_var.reset(token)
