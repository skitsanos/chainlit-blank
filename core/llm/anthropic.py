import logging
from typing import List, Dict, Any, Union, Optional

from anthropic import AsyncAnthropic

from core.llm.tool_handling import prepare_tools_for_api
from core.llm.types import Message, LLMResponse

logger = logging.getLogger(__name__)


async def handle_anthropic_api(
        client: AsyncAnthropic,
        user_input: Union[str, List[Message]],
        model: str,
        instructions: Optional[str],
        tools: Optional[List[Dict[str, Any]]],
        temperature: float,
        max_tokens: int,
        **kwargs
) -> LLMResponse:
    """Handle interactions with Anthropic's Claude models."""
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
        # Tools should already be in the correct format for Anthropic
        # anthropic_params["tools"] = tools
        anthropic_params["tools"] = prepare_tools_for_api(tools, 'anthropic')
        # Use auto tool choice - this is the proper format based on docs
        anthropic_params["tool_choice"] = {"type": "auto"}
        logger.info(f"Sending {len(tools)} tools to Anthropic API")

    # Log parameters for debugging
    logger.info(f"Anthropic API parameters: {anthropic_params}")

    # Call Anthropic API
    try:
        message = await client.messages.create(**anthropic_params)
        print(message)

        # Process tool use in the response if present
        final_text = ""
        if hasattr(message, 'content') and message.content:
            for content_block in message.content:
                if content_block.type == "text":
                    final_text += content_block.text

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
            if hasattr(message, 'content') and message.content:
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
