import random
import time
from typing import List, Optional, Union

import requests

from letta.constants import CLI_WARNING_PREFIX
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
def create(
    # agent_state: AgentState,
    llm_config: LLMConfig,
    messages: List[Message],
    user_id: Optional[str] = None,  # option UUID to associate request with
    functions: Optional[list] = None,
    functions_python: Optional[dict] = None,
    function_call: str = "auto",
    # hint
    first_message: bool = False,
    # use tool naming?
    # if false, will use deprecated 'functions' style
    use_tool_naming: bool = True,
    # streaming?
    stream: bool = False,
    stream_interface: Optional[Union[AgentRefreshStreamingInterface, AgentChunkStreamingInterface]] = None,
    max_tokens: Optional[int] = None,
    model_settings: Optional[dict] = None,  # TODO: eventually pass from server
) -> ChatCompletionResponse:
    """Return response to chat completion with backoff"""
    from letta.utils import printd

    if not model_settings:
        from letta.settings import model_settings

        model_settings = model_settings

    printd(f"Using model {llm_config.model_endpoint_type}, endpoint: {llm_config.model_endpoint}")

    if function_call and not functions:
        printd("unsetting function_call because functions is None")
        function_call = None

    # openai
    if llm_config.model_endpoint_type == "openai":
        if model_settings.openai_api_key is None and llm_config.model_endpoint == "https://api.openai.com/v1":
            # only is a problem if we are *not* using an openai proxy
            raise ValueError(f"OpenAI key is missing from letta config file")

        data = build_openai_chat_completions_request(llm_config, messages, user_id, functions, function_call, use_tool_naming, max_tokens)
        if stream:  # Client requested token streaming
            data.stream = True
            assert isinstance(stream_interface, AgentChunkStreamingInterface) or isinstance(
                stream_interface, AgentRefreshStreamingInterface
            ), type(stream_interface)
            response = openai_chat_completions_process_stream(
                url=llm_config.model_endpoint,  # https://api.openai.com/v1 -> https://api.openai.com/v1/chat/completions
                api_key=model_settings.openai_api_key,
                chat_completion_request=data,
                stream_interface=stream_interface,
            )
        else:  # Client did not request token streaming (expect a blocking backend response)
            data.stream = False
            if isinstance(stream_interface, AgentChunkStreamingInterface):
                stream_interface.stream_start()
            try:
                response = openai_chat_completions_request(
                    url=llm_config.model_endpoint,  # https://api.openai.com/v1 -> https://api.openai.com/v1/chat/completions
                    api_key=model_settings.openai_api_key,
                    chat_completion_request=data,
                )
            finally:
                if isinstance(stream_interface, AgentChunkStreamingInterface):
                    stream_interface.stream_end()

        if llm_config.put_inner_thoughts_in_kwargs:
            response = unpack_all_inner_thoughts_from_kwargs(response=response, inner_thoughts_key=INNER_THOUGHTS_KWARG)

        return response

    # azure
    elif llm_config.model_endpoint_type == "azure":
        if stream:
            raise NotImplementedError(f"Streaming not yet implemented for {llm_config.model_endpoint_type}")

        if model_settings.azure_api_key is None:
            raise ValueError(f"Azure API key is missing. Did you set AZURE_API_KEY in your env?")

        if model_settings.azure_base_url is None:
            raise ValueError(f"Azure base url is missing. Did you set AZURE_BASE_URL in your env?")

        if model_settings.azure_api_version is None:
            raise ValueError(f"Azure API version is missing. Did you set AZURE_API_VERSION in your env?")

        # Set the llm config model_endpoint from model_settings
        # For Azure, this model_endpoint is required to be configured via env variable, so users don't need to provide it in the LLM config
        llm_config.model_endpoint = model_settings.azure_base_url
        chat_completion_request = build_openai_chat_completions_request(
            llm_config, messages, user_id, functions, function_call, use_tool_naming, max_tokens
        )

        response = azure_openai_chat_completions_request(
            model_settings=model_settings,
            llm_config=llm_config,
            api_key=model_settings.azure_api_key,
            chat_completion_request=chat_completion_request,
        )

        if llm_config.put_inner_thoughts_in_kwargs:
            response = unpack_all_inner_thoughts_from_kwargs(response=response, inner_thoughts_key=INNER_THOUGHTS_KWARG)

        return response

    elif llm_config.model_endpoint_type == "google_ai":
        if stream:
            raise NotImplementedError(f"Streaming not yet implemented for {llm_config.model_endpoint_type}")
        if not use_tool_naming:
            raise NotImplementedError("Only tool calling supported on Google AI API requests")

        if functions is not None:
            tools = [{"type": "function", "function": f} for f in functions]
            tools = [Tool(**t) for t in tools]
            tools = convert_tools_to_google_ai_format(tools, inner_thoughts_in_kwargs=llm_config.put_inner_thoughts_in_kwargs)
        else:
            tools = None

        return google_ai_chat_completions_request(
            base_url=llm_config.model_endpoint,
            model=llm_config.model,
            api_key=model_settings.gemini_api_key,
            # see structure of payload here: https://ai.google.dev/docs/function_calling
            data=dict(
                contents=[m.to_google_ai_dict() for m in messages],
                tools=tools,
            ),
            inner_thoughts_in_kwargs=llm_config.put_inner_thoughts_in_kwargs,
        )

    elif llm_config.model_endpoint_type == "anthropic":
        if stream:
            raise NotImplementedError(f"Streaming not yet implemented for {llm_config.model_endpoint_type}")
        if not use_tool_naming:
            raise NotImplementedError("Only tool calling supported on Anthropic API requests")

        return anthropic_chat_completions_request(
            url=llm_config.model_endpoint,
            api_key=model_settings.anthropic_api_key,
            data=ChatCompletionRequest(
                model=llm_config.model,
                messages=[cast_message_to_subtype(m.to_openai_dict()) for m in messages],
                tools=[{"type": "function", "function": f} for f in functions] if functions else None,
                # tool_choice=function_call,
                # user=str(user_id),
                # NOTE: max_tokens is required for Anthropic API
                max_tokens=1024,  # TODO make dynamic
            ),
        )

    # elif llm_config.model_endpoint_type == "cohere":
    #     if stream:
    #         raise NotImplementedError(f"Streaming not yet implemented for {llm_config.model_endpoint_type}")
    #     if not use_tool_naming:
    #         raise NotImplementedError("Only tool calling supported on Cohere API requests")
    #
    #     if functions is not None:
    #         tools = [{"type": "function", "function": f} for f in functions]
    #         tools = [Tool(**t) for t in tools]
    #     else:
    #         tools = None
    #
    #     return cohere_chat_completions_request(
    #         # url=llm_config.model_endpoint,
    #         url="https://api.cohere.ai/v1",  # TODO
    #         api_key=os.getenv("COHERE_API_KEY"),  # TODO remove
    #         chat_completion_request=ChatCompletionRequest(
    #             model="command-r-plus",  # TODO
    #             messages=[cast_message_to_subtype(m.to_openai_dict()) for m in messages],
    #             tools=tools,
    #             tool_choice=function_call,
    #             # user=str(user_id),
    #             # NOTE: max_tokens is required for Anthropic API
    #             # max_tokens=1024,  # TODO make dynamic
    #         ),
    #     )

    elif llm_config.model_endpoint_type == "groq":
        if stream:
            raise NotImplementedError(f"Streaming not yet implemented for Groq.")

        if model_settings.groq_api_key is None and llm_config.model_endpoint == "https://api.groq.com/openai/v1/chat/completions":
            raise ValueError(f"Groq key is missing from letta config file")

        # force to true for groq, since they don't support 'content' is non-null
        if llm_config.put_inner_thoughts_in_kwargs:
            functions = add_inner_thoughts_to_functions(
                functions=functions,
                inner_thoughts_key=INNER_THOUGHTS_KWARG,
                inner_thoughts_description=INNER_THOUGHTS_KWARG_DESCRIPTION,
            )

        tools = [{"type": "function", "function": f} for f in functions] if functions is not None else None
        data = ChatCompletionRequest(
            model=llm_config.model,
            messages=[m.to_openai_dict(put_inner_thoughts_in_kwargs=llm_config.put_inner_thoughts_in_kwargs) for m in messages],
            tools=tools,
            tool_choice=function_call,
            user=str(user_id),
        )

        # https://console.groq.com/docs/openai
        # "The following fields are currently not supported and will result in a 400 error (yikes) if they are supplied:"
        assert data.top_logprobs is None
        assert data.logit_bias is None
        assert data.logprobs == False
        assert data.n == 1
        # They mention that none of the messages can have names, but it seems to not error out (for now)

        data.stream = False
        if isinstance(stream_interface, AgentChunkStreamingInterface):
            stream_interface.stream_start()
        try:
            # groq uses the openai chat completions API, so this component should be reusable
            response = openai_chat_completions_request(
                url=llm_config.model_endpoint,
                api_key=model_settings.groq_api_key,
                chat_completion_request=data,
            )
        finally:
            if isinstance(stream_interface, AgentChunkStreamingInterface):
                stream_interface.stream_end()

        if llm_config.put_inner_thoughts_in_kwargs:
            response = unpack_all_inner_thoughts_from_kwargs(response=response, inner_thoughts_key=INNER_THOUGHTS_KWARG)

        return response

    # local model
    else:
        if stream:
            raise NotImplementedError(f"Streaming not yet implemented for {llm_config.model_endpoint_type}")
        return get_chat_completion(
            model=llm_config.model,
            messages=messages,
            functions=functions,
            functions_python=functions_python,
            function_call=function_call,
            context_window=llm_config.context_window,
            endpoint=llm_config.model_endpoint,
            endpoint_type=llm_config.model_endpoint_type,
            wrapper=llm_config.model_wrapper,
            user=str(user_id),
            # hint
            first_message=first_message,
            # auth-related
            auth_type=model_settings.openllm_auth_type,
            auth_key=model_settings.openllm_api_key,
        )
