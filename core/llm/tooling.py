import inspect
import logging
from typing import Set

logger = logging.getLogger(__name__)

# Types of tools that are handled internally by OpenAI
INTERNAL_TOOL_TYPES = {"file_search", "web_search", "web_search_preview", "code_interpreter", "retrieval"}

# Mapping Python types to JSON schema types
TYPE_MAP = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object"
}


def llm_tool(func):
    """
    Generic decorator to register a function as a tool for multiple LLM providers.

    This decorator inspects the function signature and creates schema definitions
    for OpenAI, Anthropic, and potentially other LLM providers.

    Args:
        func: The function to register as a tool

    Returns:
        The decorated function with LLM-specific tool schemas attached
    """
    signature = inspect.signature(func)

    # Common properties for all LLM schemas
    func_name = func.__name__
    func_description = func.__doc__ or f"Function {func_name}"

    # Build parameters schema
    parameters = {
        "type": "object",
        "properties": {},
        "required": []
    }

    for name, param in signature.parameters.items():
        param_type = TYPE_MAP.get(param.annotation.__origin__, "string") if hasattr(
            param.annotation, '__origin__') else TYPE_MAP.get(param.annotation, "string")

        if hasattr(param.annotation, '__metadata__'):
            description = param.annotation.__metadata__[0]
            parameters["properties"][name] = {
                "type": param_type,
                "description": description
            }
            parameters["required"].append(name)
        else:
            parameters["properties"][name] = {
                "type": param_type
            }
            # Add to required if no default value
            if param.default is param.empty:
                parameters["required"].append(name)

    # Create OpenAI tool schema
    func.openai_tool = {
        "type": "function",
        "name": func_name,
        "function": {
            "description": func_description,
            "parameters": parameters
        }
    }

    # Create Anthropic tool schema
    func.anthropic_tool = {
        "name": func_name,
        "description": func_description,
        "input_schema": {
            "type": "object",
            "properties": parameters["properties"],
            "required": parameters["required"]
        }
    }

    # Generic tool schema (used by the registry for execution)
    func.tool = {
        "name": func_name,
        "description": func_description,
        "parameters": parameters
    }

    return func


class ToolRegistry:
    def __init__(self, internal_tool_types: Set[str] = None):
        self._tools = {}
        self._schema_cache = {}  # Cache for schemas by provider
        self._cache_valid = {}  # Track if cache is valid for each provider
        self._internal_tool_types = internal_tool_types or INTERNAL_TOOL_TYPES.copy()

    def register(self, name, func):
        """Register a tool function with its schema."""
        if not hasattr(func, 'tool'):
            raise ValueError(f"Function {name} does not have a 'tool' attribute")

        self._tools[name] = func

        # Invalidate all schema caches when a new tool is registered
        for provider in self._cache_valid:
            self._cache_valid[provider] = False

        return self

    def unregister(self, name):
        """Unregister a tool function."""
        if name in self._tools:
            del self._tools[name]

            # Invalidate all schema caches when a tool is removed
            for provider in self._cache_valid:
                self._cache_valid[provider] = False

        return self

    def is_internal_tool(self, tool_type: str) -> bool:
        """
        Check if a tool type is handled internally by the LLM provider (not by our registry).

        Args:
            tool_type: The type of the tool to check

        Returns:
            True if the tool type is handled internally, False otherwise
        """
        return tool_type in self._internal_tool_types

    def add_internal_tool_type(self, tool_type: str):
        """
        Register a tool type as being handled internally by the LLM provider.

        Args:
            tool_type: The type of internal tool
        """
        self._internal_tool_types.add(tool_type)

    def get_internal_tool_types(self) -> Set[str]:
        """
        Get all registered internal tool types.

        Returns:
            Set of internal tool types
        """
        return self._internal_tool_types.copy()

    def get(self, name):
        """Get a tool function by name."""
        if name not in self._tools:
            raise KeyError(f"Tool '{name}' not registered")
        return self._tools[name]

    def get_schemas(self, provider="openai"):
        """
        Get all tool schemas for a specific provider.
        Uses cached schemas if available and valid.
        """
        # Initialize cache for provider if not exists
        if provider not in self._schema_cache:
            self._schema_cache[provider] = []
            self._cache_valid[provider] = False

        # Build schema cache if invalid
        if not self._cache_valid.get(provider, False):
            self._build_schema_cache(provider)
            self._cache_valid[provider] = True

        return self._schema_cache[provider]

    def _build_schema_cache(self, provider):
        """Build the schema cache for a specific provider."""
        schemas = []

        for name, func in self._tools.items():
            if provider == "openai" and hasattr(func, 'openai_tool'):
                schemas.append(func.openai_tool)
            elif provider == "anthropic" and hasattr(func, 'anthropic_tool'):
                schemas.append(func.anthropic_tool)
            else:
                # Always add the generic tool schema if specific provider schema not available
                schemas.append(func.tool)

        self._schema_cache[provider] = schemas
        logger.debug(f"Built schema cache for provider {provider} with {len(schemas)} tools")

    def get_all(self):
        """Get all registered tools."""
        return self._tools

    def get_names(self):
        """Get all registered tool names."""
        return list(self._tools.keys())

    def has_tool(self, name):
        """Check if a tool is registered."""
        return name in self._tools

    def clear_cache(self, provider=None):
        """
        Clear schema cache for a specific provider or all providers.

        Args:
            provider: Optional provider name, if None clears all caches
        """
        if provider:
            if provider in self._schema_cache:
                self._schema_cache[provider] = []
                self._cache_valid[provider] = False
        else:
            self._schema_cache = {}
            self._cache_valid = {}

        logger.debug(f"Cleared schema cache for {provider if provider else 'all providers'}")

    async def execute_tool(self, name, args):
        """
        Execute a tool by name with the given arguments.
        Supports both synchronous and asynchronous tool functions.

        Args:
            name: Tool name
            args: Arguments to pass to the tool function

        Returns:
            Result from the tool function

        Raises:
            KeyError: If tool is not registered
            ValueError: If tool is an internal tool that should be handled by the LLM provider
            Exception: If tool execution fails
        """
        if not self.has_tool(name):
            raise KeyError(f"Tool '{name}' not registered")

        func = self.get(name)
        logger.info(f"Executing tool {name} with args: {args}")

        try:
            # Call the function with the provided arguments
            result = func(**args)

            # If the result is a coroutine (async function), await it
            if inspect.iscoroutine(result):
                result = await result

            return result
        except Exception as e:
            logger.error(f"Error executing tool {name}: {str(e)}")
            raise