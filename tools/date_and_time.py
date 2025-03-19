from datetime import datetime
from typing import Annotated

from core.llm.tooling import llm_tool


@llm_tool
def today() -> Annotated[str, "Current date and time in ISO format"]:
    """Return the current date and time in ISO format."""
    return f"{datetime.now().isoformat()}Z"
