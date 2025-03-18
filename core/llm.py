import os
from typing import List, Optional, Dict, Any, TypedDict


class LLMResponse(TypedDict):
    """Type definition for the LLM response object."""
    text: str
    input_tokens: int
    output_tokens: int
    response_id: Optional[str]


from anthropic import Anthropic
from openai import OpenAI


class LLMClient:
    """Client for interacting with various LLM providers including OpenAI, Anthropic, and Ollama."""

    def __init__(self, base_url: Optional[str] = None, api_key: Optional[str] = None):
        """
        Initialize LLM clients.

        Args:
            base_url: Optional base URL for the OpenAI API
            api_key: Optional API key (falls back to environment variables)
        """
        self.openai_client = OpenAI(
            base_url=base_url,
            api_key=api_key or os.getenv("OPENAI_API_KEY")
        )
        self.anthropic_client = Anthropic(
            api_key=api_key or os.getenv("ANTHROPIC_API_KEY")
        )

        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY") or os.getenv("OPENAI_API_KEY")

    def response(
            self,
            user_input: str,
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
        Get a response from an LLM.

        Args:
            user_input: The user's input text
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
                return self._get_anthropic_response(
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
                return self._get_openai_response(
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

    def _get_anthropic_response(
            self,
            user_input: str,
            model: str,
            instructions: str,
            tools: Optional[List[Dict[str, Any]]],
            temperature: float,
            max_tokens: int,
            **kwargs
    ) -> LLMResponse:
        """Get response from Anthropic's Claude models."""
        message = self.anthropic_client.messages.create(
            model=model,
            messages=[
                {"role": "system", "content": instructions},
                {"role": "user", "content": user_input}
            ],
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

    def _get_openai_response(
            self,
            user_input: str,
            model: str,
            instructions: str,
            tools: Optional[List[Dict[str, Any]]],
            temperature: float,
            max_tokens: int,
            use_responses_api: bool,
            previous_response_id: Optional[str] = None,
            **kwargs
    ) -> LLMResponse:
        """Get response from OpenAI or Ollama models using either the responses or chat completions API."""
        if use_responses_api:
            # Using the Responses API
            response_params = {
                "model": model,
                "input": user_input,
                "instructions": instructions,
                "temperature": temperature,
                "max_output_tokens": max_tokens,
                **kwargs
            }

            # Only add previous_response_id if it's provided
            if previous_response_id:
                response_params["previous_response_id"] = previous_response_id

            response = self.openai_client.responses.create(**response_params)

            return {
                "text": response.output_text,
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
                "response_id": response.id
            }
        else:
            # Using the Chat Completions API
            completion = self.openai_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": instructions},
                    {"role": "user", "content": user_input}
                ],
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


if __name__ == "__main__":
    # Example usage
    llm = LLMClient()
    try:
        # Get a response using the OpenAI Responses API
        response = llm.response("Tell me a short joke", "gpt-4o-mini")
        print(f"Response: {response['text']}")
        print(f"Tokens used: input={response['input_tokens']}, output={response['output_tokens']}")

        if response['response_id']:
            print(f"Response ID: {response['response_id']}")

            # Example of how to use the response_id for a follow-up question
            follow_up_response = llm.response(
                "Tell me another one",
                "gpt-4o-mini",
                previous_response_id=response['response_id']
            )
            print(f"\nFollow-up response: {follow_up_response['text']}")

    except Exception as e:
        print(f"Error: {e}")
