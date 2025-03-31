from typing import Optional, TypedDict, Literal


class Message(TypedDict):
    """Structure for a single message in a conversation."""
    role: Literal["user", "assistant", "system", "developer"]
    content: str


class LLMResponse(TypedDict):
    """Type definition for the LLM response object."""
    text: str
    input_tokens: int
    output_tokens: int
    response_id: Optional[str]
    sources: Optional[list[str]]
