import logging

import chainlit as cl
from chainlit.input_widget import Select, Slider

from core.commands import assistant_commands
from core.llm import LLMClient, LLMResponse

logger = logging.getLogger(__name__)

MODELS = [
    "gpt-4o",
    "gpt-4o-mini",
]


@cl.on_chat_start
async def start():
    cl.user_session.set("message_history", [])
    cl.user_session.set("default_model", 'gpt-4o-mini')
    cl.user_session.set("previous_response_id", None)

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
        ]
    ).send()

    if app_user and "identifier" in app_user:
        await cl.Message(f"Hello {app_user.identifier}").send()


@cl.on_settings_update
async def settings_update(settings):
    cl.user_session.set("default_model", settings["model"])
    cl.user_session.set("temperature", settings["temperature"])
    # cl.user_session.set("streaming", settings["Streaming"])

    await cl.Message(
        content=f"Switched to {settings['Model']} model for further conversations.",
    ).send()

    return settings


@cl.set_starters
async def set_starters():
    return [
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

    llm = LLMClient()

    llm_response: LLMResponse = llm.response(
        message.content,
        cl.user_session.get("default_model"),
        temperature=cl.user_session.get("temperature"),
        use_responses_api=True,
        previous_response_id=cl.user_session.get("previous_response_id"),
    )

    # Store the response ID in the user session
    cl.user_session.set("previous_response_id", llm_response.get("response_id"))

    await cl.Message(
        content=llm_response.get("text"),
        metadata={"tokens": llm_response.get("input_tokens") + llm_response.get("output_tokens")},
    ).send()
