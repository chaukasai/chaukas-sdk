"""
Chaukas - One-line instrumentation for agent building SDKs.

This is a namespace package that contains both:
- chaukas.spec (from chaukas-spec-client)
- chaukas.sdk (from chaukas-sdk)
"""

# Declare this as a namespace package
__path__ = __import__("pkgutil").extend_path(__path__, __name__)

# Re-export SDK functions for convenience (import chaukas.enable_chaukas())
try:
    from chaukas.sdk import (
        enable_chaukas,
        disable_chaukas,
        is_enabled,
        get_tracer,
        get_client,
        get_langchain_callback,
    )

    __all__ = [
        "enable_chaukas",
        "disable_chaukas",
        "is_enabled",
        "get_tracer",
        "get_client",
        "get_langchain_callback",
    ]
except ImportError:
    # SDK not installed yet
    pass

__version__ = "0.1.0"
