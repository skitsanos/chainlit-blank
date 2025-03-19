# Chainlit Blank Template

A starter template for building Chainlit applications with support for OpenAI and Anthropic models and tools
integration.

## Features

- **Multi-Model Support**: Integration with OpenAI and Anthropic models
- **Tool Calling**: Structured function calling with OpenAI's Responses API and Chat Completions API as well as
  Anthropic's Claude API
- **Asynchronous Client**: Built-in async support for better performance
- **Easy Tool Registry**: Register and manage custom tools
- **ChainLit UI**: Clean user interface with model selection, temperature control, and system instructions customization
- **Vector Store Integration**: Support for OpenAI vector stores

## Getting Started

### Prerequisites

- Python 3.9+
- OpenAI API key and/or Anthropic API key

### Installation

1. Clone this repository:

```bash
git clone https://github.com/skitsanos/chainlit-blank.git
cd chainlit-blank
```

1. Install dependencies:

```bash
pip install -r requirements.txt
```

1. Set up environment variables:

```bash
# Create a .env file with your API keys
echo "OPENAI_API_KEY=your_openai_key_here" > .env
echo "ANTHROPIC_API_KEY=your_anthropic_key_here" >> .env
```

### Running the Application

Start the Chainlit server:

```bash
chainlit run app.py
```

The app will be available at `http://localhost:8000`.

## Project Structure

```
.
├── app.py                       # Main ChainLit application entry point
├── core/
│   ├── __init__.py              # Core module initialization
│   ├── llm/                     # LLM client implementations
│   │   ├── __init__.py          # LLM module initialization
│   │   ├── anthropic.py         # Anthropic API handling
│   │   ├── chat_completions.py  # OpenAI Chat Completions API
│   │   ├── client.py            # Main async LLM client
│   │   ├── responses_api.py     # OpenAI Responses API
│   │   ├── tool_handling.py     # Tool processing utilities
│   │   ├── tooling.py           # Tool registry and decorators
│   │   └── types.py             # Type definitions
├── tools/
│   └── date_and_time.py         # Example tool for retrieving current date/time
└── requirements.txt             # Project dependencies
```

## Tool Integration

### Creating Custom Tools

Create your own tools by using the `@llm_tool` decorator:

```python
from core.tooling import llm_tool
from typing import Annotated

@llm_tool
def my_tool(param1: Annotated[str, "Description of param1"], 
            param2: Annotated[int, "Description of param2"]) -> str:
    """Tool description goes here."""
    # Your tool logic
    return f"Result: {param1}, {param2}"
```

### Registering Tools

Register tools with the tool registry:

```python
from core.llm.tooling import ToolRegistry
from tools.my_tools import my_tool

registry = ToolRegistry()
registry.register('my_tool', my_tool)
```

### Vector Store Integration

To use OpenAI vector stores:

1. Create a vector store in OpenAI
2. Add the vector store ID to the UI settings or set the `OPENAI_VECTOR_STORE_ID` environment variable

## Environment Variables

- `OPENAI_API_KEY`: Your OpenAI API key
- `ANTHROPIC_API_KEY`: Your Anthropic API key
- `OPENAI_VECTOR_STORE_ID`: (Optional) Default OpenAI vector store ID

## Customizing the UI

Modify the `app.py` file to customize the UI components:

- Add or remove models from the `MODELS` list
- Adjust the default temperature
- Customize system instructions
- Add new starter prompts

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

[MIT License](https://claude.ai/chat/LICENSE)
