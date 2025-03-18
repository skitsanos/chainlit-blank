import logging
from typing import List, Dict, Any, Union, Optional

from anthropic import AsyncAnthropic

from core.llm.tool_handling import prepare_tools_for_api
from core.llm.types import Message, LLMResponse
from core.tooling import ToolRegistry

logger = logging.getLogger(__name__)


async def handle_anthropic_api(
        client: AsyncAnthropic,
        tool_registry: ToolRegistry,
        user_input: Union[str, List[Message]],
        model: str,
        instructions: Optional[str],
        tools: Optional[List[Dict[str, Any]]],
        temperature: float,
        max_tokens: int,
        **kwargs
) -> LLMResponse:
    """Handle interactions with Anthropic's Claude models with advanced tool handling."""
    # Extract system prompt from messages or use provided instructions
    system_prompt = instructions or "You are a helpful assistant."

    # Prepare messages for Anthropic API (excluding system messages)
    anthropic_messages = []

    # Handle string input
    if isinstance(user_input, str):
        anthropic_messages = [
            {"role": "user", "content": user_input}
        ]
    else:
        # Handle message list - extract system message if present
        for msg in user_input:
            role = msg.get("role")
            content = msg.get("content", "")

            # Extract system message if present
            if role == "system" or role == "developer":
                if content and not instructions:  # Only override if instructions not provided
                    system_prompt = content
            elif role in ["user", "assistant"]:
                anthropic_messages.append({"role": role, "content": content})
            else:
                # Default fallback for unsupported roles
                anthropic_messages.append({"role": "user", "content": content})

    # Prepare parameters for Anthropic API
    anthropic_params = {
        "model": model,
        "messages": anthropic_messages,
        "system": system_prompt,  # Use top-level system parameter
        "temperature": temperature,
        "max_tokens": max_tokens,
        **{k: v for k, v in kwargs.items() if k not in ['tools', 'tool_choice']}  # Remove tools and tool_choice
    }

    # Only add tools if we have properly formatted ones
    if tools:
        # Prepare tools with Anthropic format
        anthropic_params["tools"] = prepare_tools_for_api(tools, 'anthropic')
        # Use auto tool choice
        anthropic_params["tool_choice"] = {"type": "auto"}
        logger.info(f"Sending {len(tools)} tools to Anthropic API")

    # Log parameters for debugging
    logger.info(f"Anthropic API parameters: {anthropic_params}")

    # Call Anthropic API
    try:
        message = await client.messages.create(**anthropic_params)

        # Process tool use in the response if present
        final_text = ""
        tool_calls = []

        for content_block in message.content:
            if content_block.type == "text":
                final_text += content_block.text
            elif content_block.type == "tool_use":
                # Process tool use blocks
                try:
                    tool_name = content_block.name
                    tool_id = content_block.id
                    tool_input = content_block.input

                    # Validate tool exists in registry
                    if not tool_registry.has_tool(tool_name):
                        logger.warning(f"Tool {tool_name} not found in registry")
                        continue

                    # Execute the tool
                    result = await tool_registry.execute_tool(tool_name, tool_input or {})

                    # Store the tool call information
                    tool_calls.append({
                        "tool_call_id": tool_id,
                        "output": str(result)
                    })
                except Exception as e:
                    logger.error(f"Error processing tool call {content_block.name}: {e}")

        # If we had tool calls, we need to make a follow-up call
        if tool_calls:
            # Modify the input to include tool outputs
            anthropic_messages.append({
                "role": "assistant",
                "content": final_text
            })

            for tool_call in tool_calls:
                anthropic_messages.append({
                    "role": "tool",
                    "content": tool_call["output"],
                    "tool_call_id": tool_call["tool_call_id"]
                })

            # Recursive call with tool outputs
            return await handle_anthropic_api(
                client=client,
                tool_registry=tool_registry,
                user_input=anthropic_messages,
                model=model,
                instructions=system_prompt,
                tools=tools,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )

        return {
            "text": final_text,
            "input_tokens": message.usage.input_tokens,
            "output_tokens": message.usage.output_tokens,
            "response_id": None  # Claude doesn't provide a response ID in the same way
        }

    except Exception as e:
        logger.error(f"Anthropic API error: {str(e)}")
        if "Extra inputs are not permitted" in str(e) or "tool_choice" in str(e):
            # If we get tool-related errors, try without tools
            logger.warning("Retrying Anthropic API call without tools")
            anthropic_params.pop("tools", None)
            anthropic_params.pop("tool_choice", None)
            message = await client.messages.create(**anthropic_params)

            final_text = ""
            if message.content:
                for content_block in message.content:
                    if content_block.type == "text":
                        final_text += content_block.text

            return {
                "text": final_text,
                "input_tokens": message.usage.input_tokens,
                "output_tokens": message.usage.output_tokens,
                "response_id": None
            }
        else:
            # Re-raise other errors
            raise