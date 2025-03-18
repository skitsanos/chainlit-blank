import os
from typing import List, Optional, Dict, Any, TypedDict, Union, Literal

# Import async clients
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI


class LLMResponse(TypedDict):
    """Type definition for the LLM response object."""
    text: str
    input_tokens: int
    output_tokens: int
    response_id: Optional[str]


# Define message structure
class Message(TypedDict):
    """Structure for a single message in a conversation."""
    role: Literal["user", "assistant", "system", "developer"]
    content: str


class AsyncLLMClient:
    """Async client for interacting with various LLM providers including OpenAI, Anthropic, and Ollama."""

    def __init__(self, base_url: Optional[str] = None, api_key: Optional[str] = None):
        """
        Initialize async LLM clients.

        Args:
            base_url: Optional base URL for the OpenAI API
            api_key: Optional API key (falls back to environment variables)
        """
        self.openai_client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key or os.getenv("OPENAI_API_KEY")
        )
        self.anthropic_client = AsyncAnthropic(
            api_key=api_key or os.getenv("ANTHROPIC_API_KEY")
        )

        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY") or os.getenv("OPENAI_API_KEY")

    async def response(
            self,
            user_input: Union[str, List[Message]],
            model: str,
            instructions: Optional[str] = None,
            tools: Optional[List[Dict[str, Any]]] = None,
            temperature: float = 0.0,
            max_tokens: int = 4096,
            use_responses_api: bool = True,
            previous_response_id: Optional[str] = None,
            **kwargs
    ) -> LLMResponse:
        """
        Get a response from an LLM asynchronously.

        Args:
            user_input: Either a string or a list of message objects with role and content
            model: Model identifier (e.g., "claude-3-opus-20240229", "gpt-4o")
            instructions: System instructions for the model
            tools: Tool definitions for function calling
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            use_responses_api: Whether to use OpenAI's responses API (vs. chat completions)
            previous_response_id: ID of the previous response (only for OpenAI Responses API)
            **kwargs: Additional parameters to pass to the API

        Returns:
            LLMResponse object containing:
                - text: The LLM's response text
                - input_tokens: Number of input tokens used
                - output_tokens: Number of output tokens used
                - response_id: ID of the response (only for OpenAI Responses API, otherwise None)

        Raises:
            ValueError: If the model name is not recognized
            Exception: If the API call fails
        """
        instructions = instructions or "You are a helpful assistant."

        try:
            if model.startswith(("claude", "anthropic")):
                return await self._get_anthropic_response(
                    user_input=user_input,
                    model=model,
                    instructions=instructions,
                    tools=tools,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **kwargs
                )
            elif any(model.startswith(prefix) for prefix in
                     ["gpt", "o1", "o3", "text-", "dall-e"]) or self.api_key == "ollama":
                return await self._get_openai_response(
                    user_input=user_input,
                    model=model,
                    instructions=instructions,
                    tools=tools,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    use_responses_api=use_responses_api,
                    previous_response_id=previous_response_id,
                    **kwargs
                )
            else:
                raise ValueError(f"Unsupported model: {model}")
        except Exception as e:
            # Re-raise with more context
            raise Exception(f"Error getting response from {model}: {str(e)}") from e

    def _prepare_messages(
            self,
            user_input: Union[str, List[Message]],
            instructions: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """
        Prepare messages based on user input type.

        Args:
            user_input: Either a string or list of Message objects
            instructions: Optional system instructions

        Returns:
            List of message dictionaries formatted for API calls
        """
        messages = []

        # Add system message if instructions are provided
        if instructions:
            messages.append({"role": "system", "content": instructions})

        # Process based on input type
        if isinstance(user_input, str):
            messages.append({"role": "user", "content": user_input})
        else:
            # Convert Message objects to the format expected by the APIs
            for msg in user_input:
                messages.append({"role": msg["role"], "content": msg["content"]})

        return messages

    async def _get_anthropic_response(
            self,
            user_input: Union[str, List[Message]],
            model: str,
            instructions: str,
            tools: Optional[List[Dict[str, Any]]],
            temperature: float,
            max_tokens: int,
            **kwargs
    ) -> LLMResponse:
        """Get response from Anthropic's Claude models asynchronously."""
        # Prepare messages for Anthropic API
        anthropic_messages = []

        # Handle string input
        if isinstance(user_input, str):
            anthropic_messages = [
                {"role": "user", "content": user_input}
            ]
        else:
            # Handle message list - map developer role to system for Claude
            for msg in user_input:
                role = "system" if msg["role"] == "developer" else msg["role"]
                if role not in ["user", "assistant", "system"]:
                    role = "user"  # Default fallback for unsupported roles
                anthropic_messages.append({"role": role, "content": msg["content"]})

        # Always ensure system message is first for Claude
        if instructions and not any(msg["role"] == "system" for msg in anthropic_messages):
            anthropic_messages.insert(0, {"role": "system", "content": instructions})

        message = await self.anthropic_client.messages.create(
            model=model,
            messages=anthropic_messages,
            tools=tools,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )

        return {
            "text": message.content[0].text,
            "input_tokens": message.usage.input_tokens,
            "output_tokens": message.usage.output_tokens,
            "response_id": None  # Claude doesn't provide a response ID in the same way
        }

    async def _get_openai_response(
            self,
            user_input: Union[str, List[Message]],
            model: str,
            instructions: str,
            tools: Optional[List[Dict[str, Any]]],
            temperature: float,
            max_tokens: int,
            use_responses_api: bool,
            previous_response_id: Optional[str] = None,
            **kwargs
    ) -> LLMResponse:
        """Get response from OpenAI or Ollama models asynchronously."""
        if use_responses_api:
            # Using the Responses API
            if isinstance(user_input, str):
                # String input for Responses API
                response_params = {
                    "model": model,
                    "input": user_input,
                    "instructions": instructions,
                    "temperature": temperature,
                    "max_output_tokens": max_tokens,
                    "tools": tools,
                    **kwargs
                }
            else:
                # Message list input for Responses API
                response_params = {
                    "model": model,
                    "input": user_input,  # OpenAI Responses API accepts message list directly
                    "temperature": temperature,
                    "max_output_tokens": max_tokens,
                    "tools": tools,
                    **kwargs
                }

                print(tools)

                # Only add instructions if no developer message is present
                if instructions and not any(msg["role"] == "developer" for msg in user_input):
                    response_params["instructions"] = instructions

            # Only add previous_response_id if it's provided
            if previous_response_id:
                response_params["previous_response_id"] = previous_response_id

            response = await self.openai_client.responses.create(**response_params)

            return {
                "text": response.output_text,
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
                "response_id": response.id
            }
        else:
            # Using the Chat Completions API
            messages = self._prepare_messages(user_input, instructions)

            completion = await self.openai_client.chat.completions.create(
                model=model,
                messages=messages,
                tools=tools,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )

            return {
                "text": completion.choices[0].message.content,
                "input_tokens": completion.usage.prompt_tokens,
                "output_tokens": completion.usage.completion_tokens,
                "response_id": None  # Chat Completions API doesn't provide a response ID
            }


# Example usage with async/await
import asyncio


async def main():
    llm = AsyncLLMClient()
    try:
        # Example 1: Simple string input
        response = await llm.response("Tell me a short joke", "gpt-4o")
        print(f"Response (string input): {response['text']}")
        print(f"Tokens: input={response['input_tokens']}, output={response['output_tokens']}")

        # Example 2: Message list input
        messages = [
            {"role": "developer", "content": "Talk like a pirate."},
            {"role": "user", "content": "Are semicolons optional in JavaScript?"}
        ]

        response2 = await llm.response(
            messages,
            "gpt-4o",
            use_responses_api=True
        )

        print(f"\nResponse (message list input): {response2['text']}")

        # Example 3: Follow-up with previous_response_id
        if response2['response_id']:
            follow_up = await llm.response(
                "What about in Python?",
                "gpt-4o",
                previous_response_id=response2['response_id']
            )
            print(f"\nFollow-up response: {follow_up['text']}")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
