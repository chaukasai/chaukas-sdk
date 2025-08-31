"""
Configuration management for Chaukas SDK.
Reads required configuration from environment variables.
"""

import os
from typing import Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class ChaukasConfig:
    """Configuration for Chaukas SDK."""
    
    # Required fields from environment
    tenant_id: str
    project_id: str
    endpoint: str
    api_key: str
    
    # Optional fields with defaults
    batch_size: int = 100
    flush_interval: float = 5.0
    timeout: float = 30.0
    
    # Optional metadata
    branch: Optional[str] = None
    tags: Optional[list] = None
    
    # Output configuration
    output_mode: str = "api"  # "api" or "file"
    output_file: Optional[str] = None
    
    @classmethod
    def from_env(cls) -> "ChaukasConfig":
        """
        Load configuration from environment variables.
        
        Required environment variables:
        - CHAUKAS_TENANT_ID
        - CHAUKAS_PROJECT_ID
        
        Required for API mode:
        - CHAUKAS_ENDPOINT
        - CHAUKAS_API_KEY
        
        Required for file mode:
        - CHAUKAS_OUTPUT_FILE
        
        Optional environment variables:
        - CHAUKAS_OUTPUT_MODE (default: "api", options: "api", "file")
        - CHAUKAS_BATCH_SIZE (default: 100)
        - CHAUKAS_FLUSH_INTERVAL (default: 5.0)
        - CHAUKAS_TIMEOUT (default: 30.0)
        - CHAUKAS_BRANCH (default: None)
        - CHAUKAS_TAGS (comma-separated, default: None)
        
        Raises:
            ValueError: If required environment variables are missing
        """
        # Required fields
        tenant_id = os.getenv("CHAUKAS_TENANT_ID")
        if not tenant_id:
            raise ValueError("CHAUKAS_TENANT_ID environment variable is required")
        
        project_id = os.getenv("CHAUKAS_PROJECT_ID")
        if not project_id:
            raise ValueError("CHAUKAS_PROJECT_ID environment variable is required")
        
        # Output mode and validation
        output_mode = os.getenv("CHAUKAS_OUTPUT_MODE", "api")
        if output_mode not in ["api", "file"]:
            raise ValueError("CHAUKAS_OUTPUT_MODE must be 'api' or 'file'")
        
        output_file = os.getenv("CHAUKAS_OUTPUT_FILE")
        endpoint = os.getenv("CHAUKAS_ENDPOINT")
        api_key = os.getenv("CHAUKAS_API_KEY")
        
        # Validate mode-specific requirements
        if output_mode == "api":
            if not endpoint:
                raise ValueError("CHAUKAS_ENDPOINT environment variable is required for API mode")
            if not api_key:
                raise ValueError("CHAUKAS_API_KEY environment variable is required for API mode")
        elif output_mode == "file":
            if not output_file:
                raise ValueError("CHAUKAS_OUTPUT_FILE environment variable is required for file mode")
            # Set defaults for file mode
            endpoint = endpoint or "file://localhost"
            api_key = api_key or "file-mode"
        
        # Optional fields
        # For file mode, use smaller batch size for more immediate writes
        default_batch_size = "1" if output_mode == "file" else "100"
        batch_size = int(os.getenv("CHAUKAS_BATCH_SIZE", default_batch_size))
        flush_interval = float(os.getenv("CHAUKAS_FLUSH_INTERVAL", "5.0"))
        timeout = float(os.getenv("CHAUKAS_TIMEOUT", "30.0"))
        
        branch = os.getenv("CHAUKAS_BRANCH")
        
        tags_str = os.getenv("CHAUKAS_TAGS")
        tags = tags_str.split(",") if tags_str else None
        
        return cls(
            tenant_id=tenant_id,
            project_id=project_id,
            endpoint=endpoint,
            api_key=api_key,
            batch_size=batch_size,
            flush_interval=flush_interval,
            timeout=timeout,
            branch=branch,
            tags=tags,
            output_mode=output_mode,
            output_file=output_file,
        )
    
    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "ChaukasConfig":
        """
        Create configuration from a dictionary.
        
        Args:
            config: Dictionary with configuration values
            
        Returns:
            ChaukasConfig instance
        """
        # Required fields
        if "tenant_id" not in config:
            raise ValueError("tenant_id is required in config")
        if "project_id" not in config:
            raise ValueError("project_id is required in config")
        if "endpoint" not in config:
            raise ValueError("endpoint is required in config")
        if "api_key" not in config:
            raise ValueError("api_key is required in config")
        
        return cls(
            tenant_id=config["tenant_id"],
            project_id=config["project_id"],
            endpoint=config["endpoint"],
            api_key=config["api_key"],
            batch_size=config.get("batch_size", 100),
            flush_interval=config.get("flush_interval", 5.0),
            timeout=config.get("timeout", 30.0),
            branch=config.get("branch"),
            tags=config.get("tags"),
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "tenant_id": self.tenant_id,
            "project_id": self.project_id,
            "endpoint": self.endpoint,
            "api_key": self.api_key,
            "batch_size": self.batch_size,
            "flush_interval": self.flush_interval,
            "timeout": self.timeout,
            "branch": self.branch,
            "tags": self.tags,
        }


# Global configuration instance
_config: Optional[ChaukasConfig] = None


def get_config() -> ChaukasConfig:
    """
    Get the global Chaukas configuration.
    
    Loads from environment on first call.
    
    Returns:
        ChaukasConfig instance
    """
    global _config
    if _config is None:
        _config = ChaukasConfig.from_env()
    return _config


def set_config(config: ChaukasConfig) -> None:
    """
    Set the global Chaukas configuration.
    
    Args:
        config: Configuration to use
    """
    global _config
    _config = config


def reset_config() -> None:
    """Reset the global configuration (useful for testing)."""
    global _config
    _config = None