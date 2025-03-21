import json
import logging
from typing import List, Dict, Any, Union, Optional

from openai import AsyncOpenAI

from core.llm.tool_handling import extract_tool_info
from core.llm.tooling import ToolRegistry
from core.llm.types import Message, LLMResponse

logger = logging.getLogger(__name__)


async def handle_responses_api(
        client: AsyncOpenAI,
        tool_registry: ToolRegistry,
        user_input: Union[str, List[Message]],
        model: str,
        instructions: Optional[str],
        tools: Optional[List[Dict[str, Any]]],
        temperature: float,
        max_tokens: int,
        previous_response_id: Optional[str] = None,
        current_tool_call_depth: int = 0,
        max_tool_call_depth: int = 3,
        **kwargs
) -> LLMResponse:
    """Handle interactions with OpenAI's Responses API including tool calling."""
    # Check if we've reached the maximum tool call depth
    if current_tool_call_depth > max_tool_call_depth:
        logger.warning(f"Maximum tool call depth ({max_tool_call_depth}) reached. Stopping recursion.")
        return {
            "text": "I've reached the maximum number of tool calls I can make for this request. Please provide more specific instructions if needed.",
            "input_tokens": 0,
            "output_tokens": 0,
            "response_id": None
        }

    # Prepare request parameters for Responses API
    if isinstance(user_input, str):
        response_params = {
            "model": model,
            "input": user_input,
            "temperature": temperature,
            "max_output_tokens": max_tokens,
            **{k: v for k, v in kwargs.items() if k not in ['tool_outputs']}  # Filter out unsupported params
        }

        # Add tools if provided
        if tools:
            response_params["tools"] = tools

        # Add instructions if provided
        if instructions:
            response_params["instructions"] = instructions

    else:
        response_params = {
            "model": model,
            "input": user_input,
            "temperature": temperature,
            "max_output_tokens": max_tokens,
            **{k: v for k, v in kwargs.items() if k not in ['tool_outputs']}  # Filter out unsupported params
        }

        # Add tools if provided
        if tools:
            response_params["tools"] = tools

        # Add instructions if provided and not in messages
        if instructions and not any(isinstance(msg, dict) and msg.get("role") == "system" for msg in user_input):
            response_params["instructions"] = instructions

    # Add previous_response_id if it's provided (for maintaining conversation context)
    if previous_response_id:
        response_params["previous_response_id"] = previous_response_id

    # Handle tool outputs if present by transforming them to a format the API expects
    if 'tool_outputs' in kwargs:
        # Check if we have any tool outputs to include
        tool_outputs = kwargs.get('tool_outputs')
        if tool_outputs:
            logger.info(f"Including {len(tool_outputs)} tool outputs in request")

            # Add each tool output to the input message array
            if isinstance(user_input, str):
                # Convert string input to message array to add tool results
                input_messages = [{"role": "user", "content": user_input}]

                # Add tool results as function_call_output type messages
                for output in tool_outputs:
                    input_messages.append({
                        "type": "function_call_output",
                        "call_id": output.get("id", output.get("tool_call_id")),
                        "output": output.get("output")
                    })

                # Update input parameter
                response_params["input"] = input_messages
            else:
                # Add tool results to existing message array
                input_messages = list(user_input)  # Make a copy to avoid modifying original

                # Add tool results as function_call_output type messages
                for output in tool_outputs:
                    input_messages.append({
                        "type": "function_call_output",
                        "call_id": output.get("id", output.get("tool_call_id")),
                        "output": output.get("output")
                    })

                # Update input parameter
                response_params["input"] = input_messages

    logger.info(f"Making Responses API call with params: {response_params}")
    response = await client.responses.create(**response_params)

    # Check for function call tool calls in the response
    # We need to filter to only process function calls, NOT internal tools like file_search
    function_calls = []
    for item in response.output:
        tool_info = extract_tool_info(item)
        if tool_info['name'] and item.type == "function_call" and not tool_registry.is_internal_tool(item.type):
            function_calls.append(item)

    # If we have function calls and haven't reached max depth, process them
    if function_calls and current_tool_call_depth < max_tool_call_depth:
        logger.info(f"Processing function calls at depth {current_tool_call_depth + 1}/{max_tool_call_depth}")

        # Process function calls with original IDs
        tool_outputs = []

        for function_call in function_calls:
            try:
                function_name = function_call.name
                arguments_json = function_call.arguments
                function_id = function_call.call_id

                logger.info(f"Handling function call: {function_name} with ID {function_id}")

                if not tool_registry.has_tool(function_name):
                    error_message = f"Error: Tool '{function_name}' not found in registry"
                    logger.error(error_message)
                    tool_outputs.append({
                        "id": function_id,
                        "output": error_message
                    })
                    continue

                try:
                    # Parse arguments
                    args = json.loads(arguments_json)

                    # Execute tool
                    result = await tool_registry.execute_tool(function_name, args)

                    # Format result
                    if isinstance(result, dict):
                        formatted_result = json.dumps(result)
                    else:
                        formatted_result = str(result)

                    # Add to tool outputs using the original ID
                    tool_outputs.append({
                        "id": function_id,
                        "output": formatted_result
                    })
                    logger.info(f"Tool processed: {function_name}")

                except json.JSONDecodeError as e:
                    error_message = f"Invalid JSON arguments for tool {function_name}: {e}"
                    logger.error(error_message)
                    tool_outputs.append({
                        "id": function_id,
                        "output": error_message
                    })
                except Exception as e:
                    error_message = f"Error executing tool {function_name}: {str(e)}"
                    logger.error(f"{error_message}\nArguments: {arguments_json}")
                    tool_outputs.append({
                        "id": function_id,
                        "output": error_message
                    })
            except Exception as e:
                error_message = f"Unexpected error processing function call: {str(e)}"
                logger.error(error_message)
                function_id = getattr(function_call, 'call_id', 'unknown')
                tool_outputs.append({
                    "id": function_id,
                    "output": error_message
                })

        # If we have tool outputs, make a recursive call to continue the conversation
        if tool_outputs:
            logger.info(f"Continuing conversation with {len(tool_outputs)} tool outputs")

            # Make a follow-up call to continue the conversation with tool outputs
            # Important: Pass tool_outputs in kwargs to let the function handle them properly
            return await handle_responses_api(
                client=client,
                tool_registry=tool_registry,
                user_input=user_input,  # Keep the original input
                model=model,
                instructions=instructions,
                tools=tools,
                temperature=temperature,
                max_tokens=max_tokens,
                previous_response_id=response.id,  # Important: use the response ID for continuity
                current_tool_call_depth=current_tool_call_depth + 1,
                max_tool_call_depth=max_tool_call_depth,
                tool_outputs=tool_outputs,  # Add tool outputs via kwargs
                **{k: v for k, v in kwargs.items() if k != 'tool_outputs'}  # Pass through other kwargs
            )

    # If no tool calls or max depth reached, return the response as is
    return {
        "text": response.output_text,
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
        "response_id": response.id
    }
