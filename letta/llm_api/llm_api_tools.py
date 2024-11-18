import random
import time
from typing import List, Optional, Union

import requests

from letta.constants import CLI_WARNING_PREFIX
from letta.interface import RequestContext
from letta.llm_api.anthropic import anthropic_chat_completions_request
from letta.llm_api.azure_openai import azure_openai_chat_completions_request
from letta.llm_api.google_ai import (
    convert_tools_to_google_ai_format,
    google_ai_chat_completions_request,
)
from letta.llm_api.helpers import (
    add_inner_thoughts_to_functions,
    unpack_all_inner_thoughts_from_kwargs,
)
from letta.llm_api.openai import (
    build_openai_chat_completions_request,
    openai_chat_completions_process_stream,
    openai_chat_completions_request,
)
from letta.local_llm.chat_completion_proxy import get_chat_completion
from letta.local_llm.constants import (
    INNER_THOUGHTS_KWARG,
    INNER_THOUGHTS_KWARG_DESCRIPTION,
)
from letta.schemas.llm_config import LLMConfig
from letta.schemas.message import Message
from letta.schemas.openai.chat_completion_request import (
    ChatCompletionRequest,
    Tool,
    cast_message_to_subtype,
)
from letta.schemas.openai.chat_completion_response import ChatCompletionResponse
from letta.streaming_interface import (
    AgentChunkStreamingInterface,
    AgentRefreshStreamingInterface,
)

LLM_API_PROVIDER_OPTIONS = ["openai", "azure", "anthropic", "google_ai", "cohere", "local", "groq"]


def retry_with_exponential_backoff(
    func,
    initial_delay: float = 1,
    exponential_base: float = 2,
    jitter: bool = True,
    max_retries: int = 20,
    # List of OpenAI error codes: https://github.com/openai/openai-python/blob/17ac6779958b2b74999c634c4ea4c7b74906027a/src/openai/_client.py#L227-L250
    # 429 = rate limit
    error_codes: tuple = (429,),
):
    """Retry a function with exponential backoff."""

    def wrapper(*args, **kwargs):
        pass

        # Initialize variables
        num_retries = 0
        delay = initial_delay

        # Loop until a successful response or max_retries is hit or an exception is raised
        while True:
            try:
                return func(*args, **kwargs)

            except requests.exceptions.HTTPError as http_err:

                if not hasattr(http_err, "response") or not http_err.response:
                    raise

                # Retry on specified errors
                if http_err.response.status_code in error_codes:
                    # Increment retries
                    num_retries += 1

                    # Check if max retries has been reached
                    if num_retries > max_retries:
                        raise Exception(f"Maximum number of retries ({max_retries}) exceeded.")

                    # Increment the delay
                    delay *= exponential_base * (1 + jitter * random.random())

                    # Sleep for the delay
                    # printd(f"Got a rate limit error ('{http_err}') on LLM backend request, waiting {int(delay)}s then retrying...")
                    print(
                        f"{CLI_WARNING_PREFIX}Got a rate limit error ('{http_err}') on LLM backend request, waiting {int(delay)}s then retrying..."
                    )
                    time.sleep(delay)
                else:
                    # For other HTTP errors, re-raise the exception
                    raise

            # Raise exceptions for any errors not specified
            except Exception as e:
                raise e

    return wrapper


@retry_with_exponential_backoff
def create(request_context: RequestContext) -> ChatCompletionResponse:
    """Return response to chat completion with backoff."""
    from letta.utils import printd

    llm_config = request_context.llm_config
    model_settings = request_context.model_settings or get_default_model_settings()

    printd(f"Using model {llm_config.model_endpoint_type}, endpoint: {llm_config.model_endpoint}")

    if request_context.function_call and not request_context.functions:
        printd("Unsetting function_call because functions is None")
        request_context.function_call = None

    def handle_openai():
        check_api_key(model_settings.openai_api_key, llm_config.model_endpoint, "OpenAI")
        data = build_openai_chat_completions_request(
            llm_config, request_context.messages, request_context.user_id, request_context.functions,
            request_context.function_call, request_context.use_tool_naming, request_context.max_tokens
        )
        data.stream = request_context.stream
        return process_openai_request(data, model_settings.openai_api_key, request_context)

    def handle_azure():
        check_streaming_support(request_context.stream, "Azure")
        check_azure_settings(model_settings)
        llm_config.model_endpoint = model_settings.azure_base_url
        chat_completion_request = build_openai_chat_completions_request(
            llm_config, request_context.messages, request_context.user_id, request_context.functions,
            request_context.function_call, request_context.use_tool_naming, request_context.max_tokens
        )
        return azure_openai_chat_completions_request(
            model_settings=model_settings,
            llm_config=llm_config,
            api_key=model_settings.azure_api_key,
            chat_completion_request=chat_completion_request,
        )

    def handle_google_ai():
        check_streaming_support(request_context.stream, "Google AI")
        check_tool_naming_support(request_context.use_tool_naming, "Google AI")
        tools = convert_tools_to_google_ai_format(
            [{"type": "function", "function": f} for f in request_context.functions] if request_context.functions else None,
            inner_thoughts_in_kwargs=llm_config.put_inner_thoughts_in_kwargs
        )
        return google_ai_chat_completions_request(
            base_url=llm_config.model_endpoint,
            model=llm_config.model,
            api_key=model_settings.gemini_api_key,
            data=dict(contents=[m.to_google_ai_dict() for m in request_context.messages], tools=tools),
            inner_thoughts_in_kwargs=llm_config.put_inner_thoughts_in_kwargs,
        )

    def handle_anthropic():
        check_streaming_support(request_context.stream, "Anthropic")
        check_tool_naming_support(request_context.use_tool_naming, "Anthropic")
        return anthropic_chat_completions_request(
            url=llm_config.model_endpoint,
            api_key=model_settings.anthropic_api_key,
            data=ChatCompletionRequest(
                model=llm_config.model,
                messages=[cast_message_to_subtype(m.to_openai_dict()) for m in request_context.messages],
                tools=[{"type": "function", "function": f} for f in request_context.functions] if request_context.functions else None,
                max_tokens=1024,
            ),
        )

    def handle_groq():
        check_streaming_support(request_context.stream, "Groq")
        check_api_key(model_settings.groq_api_key, llm_config.model_endpoint, "Groq")
        if llm_config.put_inner_thoughts_in_kwargs:
            request_context.functions = add_inner_thoughts_to_functions(
                functions=request_context.functions,
                inner_thoughts_key=INNER_THOUGHTS_KWARG,
                inner_thoughts_description=INNER_THOUGHTS_KWARG_DESCRIPTION,
            )
        tools = [{"type": "function", "function": f} for f in request_context.functions] if request_context.functions else None
        data = ChatCompletionRequest(
            model=llm_config.model,
            messages=[m.to_openai_dict(put_inner_thoughts_in_kwargs=llm_config.put_inner_thoughts_in_kwargs) for m in request_context.messages],
            tools=tools,
            tool_choice=request_context.function_call,
            user=str(request_context.user_id),
        )
        return process_groq_request(data, model_settings.groq_api_key, request_context)

    def handle_local():
        check_streaming_support(request_context.stream, "local models")
        return get_chat_completion(
            model=llm_config.model,
            messages=request_context.messages,
            functions=request_context.functions,
            functions_python=request_context.functions_python,
            function_call=request_context.function_call,
            context_window=llm_config.context_window,
            endpoint=llm_config.model_endpoint,
            endpoint_type=llm_config.model_endpoint_type,
            wrapper=llm_config.model_wrapper,
            user=str(request_context.user_id),
            first_message=request_context.first_message,
            auth_type=model_settings.openllm_auth_type,
            auth_key=model_settings.openllm_api_key,
        )

    handlers = {
        "openai": handle_openai,
        "azure": handle_azure,
        "google_ai": handle_google_ai,
        "anthropic": handle_anthropic,
        "groq": handle_groq,
        "local": handle_local,
    }

    handler = handlers.get(llm_config.model_endpoint_type)
    if handler:
        return handler()
    else:
        raise NotImplementedError(f"Model endpoint type '{llm_config.model_endpoint_type}' is not supported.")

def get_default_model_settings():
    from letta.settings import model_settings
    return model_settings

def check_api_key(api_key, endpoint, provider_name):
    if api_key is None and endpoint.startswith("https://api.openai.com/v1"):
        raise ValueError(f"{provider_name} key is missing from letta config file")

def check_streaming_support(stream, provider_name):
    if stream:
        raise NotImplementedError(f"Streaming not yet implemented for {provider_name}")

def check_tool_naming_support(use_tool_naming, provider_name):
    if not use_tool_naming:
        raise NotImplementedError(f"Only tool calling supported on {provider_name} API requests")

def check_azure_settings(model_settings):
    if not all([model_settings.azure_api_key, model_settings.azure_base_url, model_settings.azure_api_version]):
        raise ValueError("Azure API key, base URL, or version is missing. Check your environment variables.")

def process_openai_request(data, api_key, request_context):
    if request_context.stream:
        assert isinstance(request_context.stream_interface, (AgentChunkStreamingInterface, AgentRefreshStreamingInterface))
        return openai_chat_completions_process_stream(
            url=request_context.llm_config.model_endpoint,
            api_key=api_key,
            chat_completion_request=data,
            stream_interface=request_context.stream_interface,
        )
    else:
        if isinstance(request_context.stream_interface, AgentChunkStreamingInterface):
            request_context.stream_interface.stream_start()
        try:
            return openai_chat_completions_request(
                url=request_context.llm_config.model_endpoint,
                api_key=api_key,
                chat_completion_request=data,
            )
        finally:
            if isinstance(request_context.stream_interface, AgentChunkStreamingInterface):
                request_context.stream_interface.stream_end()

def process_groq_request(data, api_key, request_context):
    assert data.top_logprobs is None
    assert data.logit_bias is None
    assert data.logprobs == False
    assert data.n == 1
    data.stream = False
    if isinstance(request_context.stream_interface, AgentChunkStreamingInterface):
        request_context.stream_interface.stream_start()
    try:
        return openai_chat_completions_request(
            url=request_context.llm_config.model_endpoint,
            api_key=api_key,
            chat_completion_request=data,
        )
    finally:
        if isinstance(request_context.stream_interface, AgentChunkStreamingInterface):
            request_context.stream_interface.stream_end()