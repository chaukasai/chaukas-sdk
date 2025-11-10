"""
Agent ID mapping strategies for different frameworks.

Each framework has different ways of identifying agents:
- OpenAI: Uses agent.name as ID
- CrewAI: Has agent.id (UUID4) and agent.role
- Google ADK: Has agent._id or agent.name
"""

from typing import Any, Callable, Dict, Optional, Tuple


class AgentMapper:
    """Maps agent objects from different frameworks to standard agent_id and agent_name."""

    @staticmethod
    def map_openai_agent(agent: Any) -> Tuple[str, str]:
        """
        Map OpenAI agent to agent_id and agent_name.

        OpenAI agents don't have built-in IDs, so we use name for both.

        Args:
            agent: OpenAI agent instance

        Returns:
            Tuple of (agent_id, agent_name)
        """
        agent_name = getattr(agent, "name", "unknown")
        agent_id = agent_name  # Use name as ID since no built-in ID
        return agent_id, agent_name

    @staticmethod
    def map_crewai_agent(agent: Any, crew: Optional[Any] = None) -> Tuple[str, str]:
        """
        Map CrewAI agent to agent_id and agent_name.

        CrewAI agents have:
        - id: UUID4 auto-generated field
        - role: The agent's role
        - Crew optionally has a name

        Args:
            agent: CrewAI agent instance
            crew: Optional crew instance the agent belongs to

        Returns:
            Tuple of (agent_id, agent_name)
        """
        # Use the built-in UUID4 id
        agent_id = str(getattr(agent, "id", None) or getattr(agent, "role", "unknown"))

        # Build agent name from crew.name and agent.role
        agent_role = getattr(agent, "role", "unknown")
        if crew:
            crew_name = getattr(crew, "name", None)
            if crew_name and crew_name != "crew":  # "crew" is the default
                agent_name = f"{crew_name}.{agent_role}"
            else:
                agent_name = agent_role
        else:
            agent_name = agent_role

        return agent_id, agent_name

    @staticmethod
    def map_google_adk_agent(agent: Any) -> Tuple[str, str]:
        """
        Map Google ADK agent to agent_id and agent_name.

        Google ADK agents have:
        - _id: Internal ID (optional)
        - name: Agent name

        Args:
            agent: Google ADK agent instance

        Returns:
            Tuple of (agent_id, agent_name)
        """
        agent_name = getattr(agent, "name", "unknown")

        # Try _id first, fall back to name
        agent_id = getattr(agent, "_id", None) or agent_name

        return str(agent_id), agent_name

    @staticmethod
    def map_custom_agent(
        agent: Any,
        id_extractor: Optional[Callable[[Any], str]] = None,
        name_extractor: Optional[Callable[[Any], str]] = None,
    ) -> Tuple[str, str]:
        """
        Map custom agent using provided extractors.

        Args:
            agent: Custom agent instance
            id_extractor: Function to extract agent ID
            name_extractor: Function to extract agent name

        Returns:
            Tuple of (agent_id, agent_name)
        """
        if id_extractor:
            agent_id = id_extractor(agent)
        else:
            # Default: try common attribute names
            agent_id = (
                getattr(agent, "id", None)
                or getattr(agent, "_id", None)
                or getattr(agent, "agent_id", None)
                or getattr(agent, "name", None)
                or str(id(agent))  # Last resort: Python object ID
            )

        if name_extractor:
            agent_name = name_extractor(agent)
        else:
            # Default: try common attribute names
            agent_name = (
                getattr(agent, "name", None)
                or getattr(agent, "agent_name", None)
                or getattr(agent, "role", None)
                or getattr(agent, "title", None)
                or "unknown"
            )

        return str(agent_id), agent_name

    @staticmethod
    def detect_and_map(
        agent: Any, framework_hint: Optional[str] = None, **kwargs
    ) -> Tuple[str, str]:
        """
        Detect framework and map agent to ID and name.

        Args:
            agent: Agent instance from any framework
            framework_hint: Optional hint about which framework ("openai", "crewai", "google_adk")
            **kwargs: Additional arguments (e.g., crew for CrewAI)

        Returns:
            Tuple of (agent_id, agent_name)
        """
        # If hint provided, use specific mapper
        if framework_hint:
            if framework_hint.lower() == "openai":
                return AgentMapper.map_openai_agent(agent)
            elif framework_hint.lower() == "crewai":
                crew = kwargs.get("crew")
                return AgentMapper.map_crewai_agent(agent, crew)
            elif framework_hint.lower() in ["google_adk", "google", "adk"]:
                return AgentMapper.map_google_adk_agent(agent)

        # Try to auto-detect based on class name and attributes
        class_name = agent.__class__.__name__.lower()
        module_name = (
            agent.__class__.__module__.lower()
            if hasattr(agent.__class__, "__module__")
            else ""
        )

        # OpenAI detection
        if "openai" in module_name or "openai" in class_name:
            return AgentMapper.map_openai_agent(agent)

        # CrewAI detection
        if "crewai" in module_name or "crew" in class_name:
            crew = kwargs.get("crew")
            return AgentMapper.map_crewai_agent(agent, crew)

        # Google ADK detection
        if "google" in module_name or "adk" in module_name or "gemini" in module_name:
            return AgentMapper.map_google_adk_agent(agent)

        # Check for specific attributes as hints
        if (
            hasattr(agent, "role")
            and hasattr(agent, "goal")
            and hasattr(agent, "backstory")
        ):
            # Likely CrewAI
            crew = kwargs.get("crew")
            return AgentMapper.map_crewai_agent(agent, crew)

        if hasattr(agent, "_id") or (
            hasattr(agent, "name") and hasattr(agent, "model")
        ):
            # Likely Google ADK or OpenAI
            if hasattr(agent, "_id"):
                return AgentMapper.map_google_adk_agent(agent)
            else:
                return AgentMapper.map_openai_agent(agent)

        # Fall back to custom mapper
        return AgentMapper.map_custom_agent(
            agent, kwargs.get("id_extractor"), kwargs.get("name_extractor")
        )


# Convenience functions
def get_agent_id(agent: Any, framework: Optional[str] = None, **kwargs) -> str:
    """Get agent ID for any framework."""
    agent_id, _ = AgentMapper.detect_and_map(agent, framework, **kwargs)
    return agent_id


def get_agent_name(agent: Any, framework: Optional[str] = None, **kwargs) -> str:
    """Get agent name for any framework."""
    _, agent_name = AgentMapper.detect_and_map(agent, framework, **kwargs)
    return agent_name


def get_agent_info(
    agent: Any, framework: Optional[str] = None, **kwargs
) -> Dict[str, str]:
    """Get agent info dictionary for any framework."""
    agent_id, agent_name = AgentMapper.detect_and_map(agent, framework, **kwargs)
    return {
        "agent_id": agent_id,
        "agent_name": agent_name,
    }
