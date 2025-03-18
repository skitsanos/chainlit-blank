import logging
import os

import chainlit as cl
from chainlit.input_widget import Select, Slider, TextInput

from core.commands import assistant_commands
from core.llm import AsyncLLMClient, LLMResponse
from core.tooling import ToolRegistry
from tools.date_and_time import today

logger = logging.getLogger(__name__)

MODELS = [
    "gpt-4o",
    "gpt-4o-mini",
    "claude-3-7-sonnet-latest",
    "claude-3-5-haiku-latest"
]

# What day is today?

registry = ToolRegistry()
registry.register('today', today)


@cl.on_chat_start
async def start():
    cl.user_session.set("message_history", [])
    # cl.user_session.set("default_model", 'gpt-4o-mini')
    cl.user_session.set("default_model", 'claude-3-7-sonnet-latest')
    cl.user_session.set("previous_response_id", None)
    cl.user_session.set("temperature", 0.1)
    cl.user_session.set("vectorstore", os.environ.get("OPENAI_VECTOR_STORE_ID", None))
    cl.user_session.set("instructions", "You are a helpful assistant.")

    await cl.context.emitter.set_commands(assistant_commands)
    app_user = cl.user_session.get("user")

    # https://docs.chainlit.io/api-reference/chat-settings
    await cl.ChatSettings(
        [
            Select(
                id="model",
                label="Model",
                values=MODELS,
                # find index of the default model or set to 0
                initial_index=MODELS.index(cl.user_session.get("default_model")) if cl.user_session.get(
                    "default_model") in MODELS else 0,
            ),
            # Switch(id="Streaming", label="OpenAI - Stream Tokens", initial=True),
            Slider(
                id="temperature",
                label="Temperature",
                initial=0.1,
                min=0,
                max=2,
                step=0.1,
            ),
            TextInput(
                id="instructions",
                label="Instructions",
                placeholder="You are a helpful assistant.",
                initial="You are a helpful assistant.",
                multiline=True,
            ),
            TextInput(
                id="vectorstore",
                label="Vector Store",
                placeholder="OpenAI Vector Store ID, e.g. 'vs_123456789'",
                initial=os.environ.get("OPENAI_VECTOR_STORE_ID", None),
            )
        ]
    ).send()

    if app_user and "identifier" in app_user:
        await cl.Message(f"Hello {app_user.identifier}").send()


@cl.on_settings_update
async def settings_update(settings):
    cl.user_session.set("default_model", settings["model"])
    cl.user_session.set("temperature", settings["temperature"])
    # cl.user_session.set("streaming", settings["Streaming"])
    cl.user_session.set("instructions", settings["instructions"])
    cl.user_session.set("vectorstore", settings["vectorstore"])

    cl.logger.info(f"Settings updated: {settings}")

    await cl.Message(
        content=f"Switched to {settings['model']} model for further conversations.",
    ).send()

    return settings


@cl.set_starters
async def set_starters():
    return [
        cl.Starter(
            label="Today",
            message="What day is today?",
        ),
        cl.Starter(
            label="Pirate Joke",
            message="Tell me a pirate joke",
        ),
        cl.Starter(
            message="Haiku about the sea",
            label="Create a Haiku about the sea and the sky and the stars",
        )
    ]


@cl.on_message
async def on_message(message: cl.Message):
    if message.command == "Picture":
        # User is using the Picture command
        logger.info("User is using the Picture command")

        return

    llm = AsyncLLMClient(tool_registry=registry)

    tools = [
        *registry.get_schemas()
    ]
    vectorstore = cl.user_session.get("vectorstore")
    if vectorstore:
        all_vector_stores = vectorstore.split(",")
        # clean up the vectorstore list
        all_vector_stores = [vs.strip() for vs in all_vector_stores if vs.strip()]

        # add vector stores to the tools list
        # for vs in all_vector_stores:
        #     tools.append(
        #         {
        #             "type": "file_search",
        #             "vector_store_ids": [vs],
        #         }
        #     )

        cl.logger.info(f"Using vector store: {cl.user_session.get('vectorstore')}")

    cl.logger.info(f"Using model: {cl.user_session.get('default_model')}")
    cl.logger.info(f"Available Tools: {len(tools)}")

    thinking = cl.Step(name="Searching and Processing")
    async with thinking:
        # try:
        # Show thinking element during the entire process
        thinking_message = await cl.Message(content="Thinking...").send()

        llm_response: LLMResponse = await llm.response(
            message.content,
            cl.user_session.get("default_model"),
            instructions=cl.user_session.get("instructions"),
            temperature=cl.user_session.get("temperature"),
            tools=tools,
            use_responses_api=False,
            previous_response_id=cl.user_session.get("previous_response_id"),
        )

        # Store the response ID in the user session
        cl.user_session.set("previous_response_id", llm_response.get("response_id"))

        thinking_message.content = llm_response.get("text")
        thinking.input = message.content
        thinking.output = {"tokens": llm_response.get("input_tokens") + llm_response.get("output_tokens")}
        await thinking_message.send()

    #
