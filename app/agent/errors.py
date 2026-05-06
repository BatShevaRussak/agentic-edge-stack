"""Exception hierarchy for the agent layer."""


class AgentError(Exception):
    """Base class for failures inside the agent layer."""


class RoutingError(AgentError):
    """Router output could not be parsed in strict mode."""


class ToolExecutionError(AgentError):
    """A tool invocation failed; wraps the original cause."""
