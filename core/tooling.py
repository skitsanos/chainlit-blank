import inspect
import logging

logger = logging.getLogger(__name__)

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
    def __init__(self):
        self._tools = {}
        self._tool_schemas = {
            "openai": [],  # Tools formatted for OpenAI (both APIs)
            "anthropic": [],  # Tools formatted for Anthropic
            "generic": []  # Tools in a generic format
        }

    def register(self, name, func):
        """Register a tool function with its schema."""
        if not hasattr(func, 'tool'):
            raise ValueError(f"Function {name} does not have a 'tool' attribute")

        self._tools[name] = func

        # Add tool schemas for different providers
        if hasattr(func, 'openai_tool'):
            self._tool_schemas["openai"].append(func.openai_tool)

        if hasattr(func, 'anthropic_tool'):
            self._tool_schemas["anthropic"].append(func.anthropic_tool)

        # Always add the generic tool schema
        self._tool_schemas["generic"].append(func.tool)

        return self

    def get(self, name):
        """Get a tool function by name."""
        if name not in self._tools:
            raise KeyError(f"Tool {name} not registered")
        return self._tools[name]

    def get_schemas(self, provider="openai"):
        """Get all tool schemas for a specific provider."""
        return self._tool_schemas.get(provider, self._tool_schemas["generic"])

    def get_all(self):
        """Get all registered tools."""
        return self._tools

    def get_names(self):
        """Get all registered tool names."""
        return list(self._tools.keys())

    def has_tool(self, name):
        """Check if a tool is registered."""
        return name in self._tools

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
            Exception: If tool execution fails
        """
        if not self.has_tool(name):
            raise KeyError(f"Tool {name} not registered")

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
