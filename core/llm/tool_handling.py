import json
import logging
import uuid
from typing import Dict

logger = logging.getLogger(__name__)


def is_internal_tool(tool_type: str) -> bool:
    """Return True if this tool type is handled internally by OpenAI"""
    internal_tool_types = ["file_search", "web_search", "code_interpreter", "retrieval"]
    return tool_type in internal_tool_types


def prepare_tools_for_api(tools_list, api_type):
    """
    Format tools differently based on API type

    Args:
        tools_list: List of tool definitions
        api_type: Either 'responses' or 'completions'

    Returns:
        Properly formatted tools list for the specified API
    """
    if not tools_list:
        return None

    if api_type == 'responses':
        # Responses API accepts both function and internal tools
        return tools_list
    elif api_type == 'completions':
        # Chat Completions API only accepts function tools with specific format
        formatted_tools = []
        for tool in tools_list:
            if tool.get('type') == 'function':
                tool["function"]["name"] = tool['name']
                formatted_tools.append(tool)

        return formatted_tools if formatted_tools else None
    elif api_type == 'anthropic':
        # Anthropic API requires a different format
        formatted_tools = []
        for tool in tools_list:
            anthropic_tool = {
                "name": tool['name'],
                "description": tool["function"].get('description', ''),
                "input_schema": tool["function"].get('parameters', {}),
            }
            formatted_tools.append(anthropic_tool)

        return formatted_tools if formatted_tools else None

    return None


def create_shortened_tool_ids(tool_calls) -> Dict[str, str]:
    """Create shortened IDs for tool calls that are compatible with Chat Completions API"""
    id_mapping = {}

    for tool_call in tool_calls:
        # Generate a new UUID that's exactly 36 characters
        short_id = f"call_{str(uuid.uuid4())[:35 - 5]}"  # Keep under 40 chars
        id_mapping[getattr(tool_call, 'call_id', tool_call.id if hasattr(tool_call, 'id') else 'unknown')] = short_id
        logger.info(f"Mapped original ID {getattr(tool_call, 'call_id', 'unknown')} to shorter ID {short_id}")

    return id_mapping


async def process_function_calls(tool_registry, function_calls, id_mapping=None):
    """
    Process function calls using the tool registry with optional ID mapping for shortened IDs

    Args:
        tool_registry: Registry containing tool implementations
        function_calls: List of function call objects to process
        id_mapping: Optional mapping of original IDs to shortened IDs

    Returns:
        List of tool responses with tool_call_id and output
    """
    if id_mapping is None:
        id_mapping = create_shortened_tool_ids(function_calls)

    tool_responses = []

    for tool_call in function_calls:
        try:
            # Extract function details
            function_name = getattr(tool_call, 'name', None)
            if not function_name and hasattr(tool_call, 'function'):
                function_name = tool_call.function.name

            arguments_json = getattr(tool_call, 'arguments', None)
            if not arguments_json and hasattr(tool_call, 'function'):
                arguments_json = tool_call.function.arguments

            # Get the original ID
            original_id = getattr(tool_call, 'call_id', None)
            if not original_id and hasattr(tool_call, 'id'):
                original_id = tool_call.id

            # Use the shortened ID if available
            tool_call_id = id_mapping.get(original_id, original_id)

            logger.info(f"Handling function call: {function_name} with ID {tool_call_id}")

            if not tool_registry.has_tool(function_name):
                error_message = f"Error: Tool '{function_name}' not found in registry"
                logger.error(error_message)
                tool_responses.append({
                    "tool_call_id": tool_call_id,
                    "output": error_message
                })
                continue

            try:
                # Parse arguments JSON
                args = json.loads(arguments_json)

                # Execute the tool with the arguments
                result = await tool_registry.execute_tool(function_name, args)

                # Format the result
                if isinstance(result, dict):
                    formatted_result = json.dumps(result)
                else:
                    formatted_result = str(result)

                tool_responses.append({
                    "tool_call_id": tool_call_id,
                    "output": formatted_result
                })
                logger.info(f"Tool processed: {function_name}")
            except json.JSONDecodeError as e:
                error_message = f"Invalid JSON arguments for tool {function_name}: {e}"
                logger.error(error_message)
                tool_responses.append({
                    "tool_call_id": tool_call_id,
                    "output": error_message
                })
            except Exception as e:
                error_message = f"Error executing tool {function_name}: {str(e)}"
                logger.error(f"{error_message}\nArguments: {arguments_json}")
                tool_responses.append({
                    "tool_call_id": tool_call_id,
                    "output": error_message
                })
        except Exception as e:
            # Catch-all for any unexpected errors in tool call processing
            error_message = f"Unexpected error processing tool call: {str(e)}"
            logger.error(error_message)
            # Try to get tool_call_id if possible
            tool_call_id = id_mapping.get(getattr(tool_call, 'call_id',
                                                  getattr(tool_call, 'id', 'unknown')), 'error_id')
            tool_responses.append({
                "tool_call_id": tool_call_id,
                "output": error_message
            })

    return tool_responses


def prepare_assistant_message_with_tool_calls(content, function_calls, id_mapping):
    """
    Create an assistant message with properly formatted tool calls for Chat Completions API

    Args:
        content: Text content of the assistant message
        function_calls: List of function call objects
        id_mapping: Mapping of original IDs to shortened IDs

    Returns:
        Dict containing the assistant message with tool_calls field
    """
    assistant_message = {
        "role": "assistant",
        "content": content,
        "tool_calls": []
    }

    for tool_call in function_calls:
        # Get original ID
        original_id = getattr(tool_call, 'call_id',
                              getattr(tool_call, 'id', None))

        # Use shortened ID
        short_id = id_mapping.get(original_id, original_id)

        # Get function name and arguments
        function_name = getattr(tool_call, 'name', None)
        if not function_name and hasattr(tool_call, 'function'):
            function_name = tool_call.function.name

        arguments_json = getattr(tool_call, 'arguments', None)
        if not arguments_json and hasattr(tool_call, 'function'):
            arguments_json = tool_call.function.arguments

        assistant_message["tool_calls"].append({
            "id": short_id,
            "type": "function",
            "function": {
                "name": function_name,
                "arguments": arguments_json
            }
        })

    return assistant_message
