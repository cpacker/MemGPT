import copy
import json
import os
import random
import time
import warnings
from typing import List, Optional, Union

import requests

from memgpt.constants import CLI_WARNING_PREFIX, OPENAI_CONTEXT_WINDOW_ERROR_SUBSTRING
from memgpt.credentials import MemGPTCredentials
from memgpt.llm_api.anthropic import anthropic_chat_completions_request
from memgpt.llm_api.azure_openai import (
    MODEL_TO_AZURE_ENGINE,
    azure_openai_chat_completions_request,
)
from memgpt.llm_api.cohere import cohere_chat_completions_request
from memgpt.llm_api.google_ai import (
    convert_tools_to_google_ai_format,
    google_ai_chat_completions_request,
)
from memgpt.llm_api.openai import (
    openai_chat_completions_process_stream,
    openai_chat_completions_request,
)
from memgpt.local_llm.chat_completion_proxy import get_chat_completion
from memgpt.local_llm.constants import (
    INNER_THOUGHTS_KWARG,
    INNER_THOUGHTS_KWARG_DESCRIPTION,
)
from memgpt.schemas.enums import OptionState
from memgpt.schemas.llm_config import LLMConfig
from memgpt.schemas.message import Message
from memgpt.schemas.openai.chat_completion_request import (
    ChatCompletionRequest,
    Tool,
    cast_message_to_subtype,
)
from memgpt.schemas.openai.chat_completion_response import ChatCompletionResponse
from memgpt.streaming_interface import (
    AgentChunkStreamingInterface,
    AgentRefreshStreamingInterface,
)
from memgpt.utils import json_dumps

LLM_API_PROVIDER_OPTIONS = ["openai", "azure", "anthropic", "google_ai", "cohere", "local"]


# TODO update to use better types
def add_inner_thoughts_to_functions(
    functions: List[dict],
    inner_thoughts_key: str,
    inner_thoughts_description: str,
    inner_thoughts_required: bool = True,
    # inner_thoughts_to_front: bool = True,  TODO support sorting somewhere, probably in the to_dict?
) -> List[dict]:
    """Add an inner_thoughts kwarg to every function in the provided list"""
    # return copies
    new_functions = []

    # functions is a list of dicts in the OpenAI schema (https://platform.openai.com/docs/api-reference/chat/create)
    for function_object in functions:
        function_params = function_object["parameters"]["properties"]
        required_params = list(function_object["parameters"]["required"])

        # if the inner thoughts arg doesn't exist, add it
        if inner_thoughts_key not in function_params:
            function_params[inner_thoughts_key] = {
                "type": "string",
                "description": inner_thoughts_description,
            }

        # make sure it's tagged as required
        new_function_object = copy.deepcopy(function_object)
        if inner_thoughts_required and inner_thoughts_key not in required_params:
            required_params.append(inner_thoughts_key)
            new_function_object["parameters"]["required"] = required_params

        new_functions.append(new_function_object)

    # return a list of copies
    return new_functions


def unpack_inner_thoughts_from_kwargs(
    response: ChatCompletionResponse,
    inner_thoughts_key: str,
) -> ChatCompletionResponse:
    """Strip the inner thoughts out of the tool call and put it in the message content"""
    if len(response.choices) == 0:
        raise ValueError(f"Unpacking inner thoughts from empty response not supported")

    new_choices = []
    for choice in response.choices:
        msg = choice.message
        if msg.role == "assistant" and msg.tool_calls and len(msg.tool_calls) >= 1:
            if len(msg.tool_calls) > 1:
                warnings.warn(f"Unpacking inner thoughts from more than one tool call ({len(msg.tool_calls)}) is not supported")
            # TODO support multiple tool calls
            tool_call = msg.tool_calls[0]

            try:
                # Sadly we need to parse the JSON since args are in string format
                func_args = dict(json.loads(tool_call.function.arguments))
                if inner_thoughts_key in func_args:
                    # extract the inner thoughts
                    inner_thoughts = func_args.pop(inner_thoughts_key)

                    # replace the kwargs
                    new_choice = choice.model_copy(deep=True)
                    new_choice.message.tool_calls[0].function.arguments = json_dumps(func_args)
                    # also replace the message content
                    if new_choice.message.content is not None:
                        warnings.warn(f"Overwriting existing inner monologue ({new_choice.message.content}) with kwarg ({inner_thoughts})")
                    new_choice.message.content = inner_thoughts

                    # save copy
                    new_choices.append(new_choice)
                else:
                    warnings.warn(f"Did not find inner thoughts in tool call: {str(tool_call)}")

            except json.JSONDecodeError as e:
                warnings.warn(f"Failed to strip inner thoughts from kwargs: {e}")
                raise e

    # return an updated copy
    new_response = response.model_copy(deep=True)
    new_response.choices = new_choices
    return new_response


def is_context_overflow_error(exception: requests.exceptions.RequestException) -> bool:
    """Checks if an exception is due to context overflow (based on common OpenAI response messages)"""
    from memgpt.utils import printd

    match_string = OPENAI_CONTEXT_WINDOW_ERROR_SUBSTRING

    # Backwards compatibility with openai python package/client v0.28 (pre-v1 client migration)
    if match_string in str(exception):
        printd(f"Found '{match_string}' in str(exception)={(str(exception))}")
        return True

    # Based on python requests + OpenAI REST API (/v1)
    elif isinstance(exception, requests.exceptions.HTTPError):
        if exception.response is not None and "application/json" in exception.response.headers.get("Content-Type", ""):
            try:
                error_details = exception.response.json()
                if "error" not in error_details:
                    printd(f"HTTPError occurred, but couldn't find error field: {error_details}")
                    return False
                else:
                    error_details = error_details["error"]

                # Check for the specific error code
                if error_details.get("code") == "context_length_exceeded":
                    printd(f"HTTPError occurred, caught error code {error_details.get('code')}")
                    return True
                # Soft-check for "maximum context length" inside of the message
                elif error_details.get("message") and "maximum context length" in error_details.get("message"):
                    printd(f"HTTPError occurred, found '{match_string}' in error message contents ({error_details})")
                    return True
                else:
                    printd(f"HTTPError occurred, but unknown error message: {error_details}")
                    return False
            except ValueError:
                # JSON decoding failed
                printd(f"HTTPError occurred ({exception}), but no JSON error message.")

    # Generic fail
    else:
        return False


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
    functions_python: Optional[list] = None,
    function_call: str = "auto",
    # hint
    first_message: bool = False,
    # use tool naming?
    # if false, will use deprecated 'functions' style
    use_tool_naming: bool = True,
    # streaming?
    stream: bool = False,
    stream_inferface: Optional[Union[AgentRefreshStreamingInterface, AgentChunkStreamingInterface]] = None,
    # TODO move to llm_config?
    # if unspecified (None), default to something we've tested
    inner_thoughts_in_kwargs: OptionState = OptionState.DEFAULT,
) -> ChatCompletionResponse:
    """Return response to chat completion with backoff"""
    from memgpt.utils import printd

    printd(f"Using model {llm_config.model_endpoint_type}, endpoint: {llm_config.model_endpoint}")

    # TODO eventually refactor so that credentials are passed through

    credentials = MemGPTCredentials.load()

    if function_call and not functions:
        printd("unsetting function_call because functions is None")
        function_call = None

    # openai
    if llm_config.model_endpoint_type == "openai":

        if inner_thoughts_in_kwargs == OptionState.DEFAULT:
            # model that are known to not use `content` fields on tool calls
            inner_thoughts_in_kwargs = (
                "gpt-4o" in llm_config.model or "gpt-4-turbo" in llm_config.model or "gpt-3.5-turbo" in llm_config.model
            )
        else:
            inner_thoughts_in_kwargs = True if inner_thoughts_in_kwargs == OptionState.YES else False

        if not isinstance(inner_thoughts_in_kwargs, bool):
            warnings.warn(f"Bad type detected: {type(inner_thoughts_in_kwargs)}")
            inner_thoughts_in_kwargs = bool(inner_thoughts_in_kwargs)
        if inner_thoughts_in_kwargs:
            functions = add_inner_thoughts_to_functions(
                functions=functions,
                inner_thoughts_key=INNER_THOUGHTS_KWARG,
                inner_thoughts_description=INNER_THOUGHTS_KWARG_DESCRIPTION,
            )

        openai_message_list = [
            cast_message_to_subtype(m.to_openai_dict(put_inner_thoughts_in_kwargs=inner_thoughts_in_kwargs)) for m in messages
        ]

        # TODO do the same for Azure?
        if credentials.openai_key is None and llm_config.model_endpoint == "https://api.openai.com/v1":
            # only is a problem if we are *not* using an openai proxy
            raise ValueError(f"OpenAI key is missing from MemGPT config file")
        if use_tool_naming:
            data = ChatCompletionRequest(
                model=llm_config.model,
                messages=openai_message_list,
                tools=[{"type": "function", "function": f} for f in functions] if functions else None,
                tool_choice=function_call,
                user=str(user_id),
            )
        else:
            data = ChatCompletionRequest(
                model=llm_config.model,
                messages=openai_message_list,
                functions=functions,
                function_call=function_call,
                user=str(user_id),
            )
            # https://platform.openai.com/docs/guides/text-generation/json-mode
            # only supported by gpt-4o, gpt-4-turbo, or gpt-3.5-turbo
            if "gpt-4o" in llm_config.model or "gpt-4-turbo" in llm_config.model or "gpt-3.5-turbo" in llm_config.model:
                data.response_format = {"type": "json_object"}

        if stream:  # Client requested token streaming
            data.stream = True
            assert isinstance(stream_inferface, AgentChunkStreamingInterface) or isinstance(
                stream_inferface, AgentRefreshStreamingInterface
            ), type(stream_inferface)
            response = openai_chat_completions_process_stream(
                url=llm_config.model_endpoint,  # https://api.openai.com/v1 -> https://api.openai.com/v1/chat/completions
                api_key=credentials.openai_key,
                chat_completion_request=data,
                stream_inferface=stream_inferface,
            )
        else:  # Client did not request token streaming (expect a blocking backend response)
            data.stream = False
            if isinstance(stream_inferface, AgentChunkStreamingInterface):
                stream_inferface.stream_start()
            try:
                response = openai_chat_completions_request(
                    url=llm_config.model_endpoint,  # https://api.openai.com/v1 -> https://api.openai.com/v1/chat/completions
                    api_key=credentials.openai_key,
                    chat_completion_request=data,
                )
            finally:
                if isinstance(stream_inferface, AgentChunkStreamingInterface):
                    stream_inferface.stream_end()

        if inner_thoughts_in_kwargs:
            response = unpack_inner_thoughts_from_kwargs(response=response, inner_thoughts_key=INNER_THOUGHTS_KWARG)

        return response

    # azure
    elif llm_config.model_endpoint_type == "azure":
        if stream:
            raise NotImplementedError(f"Streaming not yet implemented for {llm_config.model_endpoint_type}")

        azure_deployment = (
            credentials.azure_deployment if credentials.azure_deployment is not None else MODEL_TO_AZURE_ENGINE[llm_config.model]
        )
        if use_tool_naming:
            data = dict(
                # NOTE: don't pass model to Azure calls, that is the deployment_id
                # model=agent_config.model,
                messages=[m.to_openai_dict() for m in messages],
                tools=[{"type": "function", "function": f} for f in functions] if functions else None,
                tool_choice=function_call,
                user=str(user_id),
            )
        else:
            data = dict(
                # NOTE: don't pass model to Azure calls, that is the deployment_id
                # model=agent_config.model,
                messages=[m.to_openai_dict() for m in messages],
                functions=functions,
                function_call=function_call,
                user=str(user_id),
            )
        return azure_openai_chat_completions_request(
            resource_name=credentials.azure_endpoint,
            deployment_id=azure_deployment,
            api_version=credentials.azure_version,
            api_key=credentials.azure_key,
            data=data,
        )

    elif llm_config.model_endpoint_type == "google_ai":
        if stream:
            raise NotImplementedError(f"Streaming not yet implemented for {llm_config.model_endpoint_type}")
        if not use_tool_naming:
            raise NotImplementedError("Only tool calling supported on Google AI API requests")

        # NOTE: until Google AI supports CoT / text alongside function calls,
        # we need to put it in a kwarg (unless we want to split the message into two)
        google_ai_inner_thoughts_in_kwarg = True

        if functions is not None:
            tools = [{"type": "function", "function": f} for f in functions]
            tools = [Tool(**t) for t in tools]
            tools = convert_tools_to_google_ai_format(tools, inner_thoughts_in_kwargs=google_ai_inner_thoughts_in_kwarg)
        else:
            tools = None

        return google_ai_chat_completions_request(
            inner_thoughts_in_kwargs=google_ai_inner_thoughts_in_kwarg,
            service_endpoint=credentials.google_ai_service_endpoint,
            model=llm_config.model,
            api_key=credentials.google_ai_key,
            # see structure of payload here: https://ai.google.dev/docs/function_calling
            data=dict(
                contents=[m.to_google_ai_dict() for m in messages],
                tools=tools,
            ),
        )

    elif llm_config.model_endpoint_type == "anthropic":
        if stream:
            raise NotImplementedError(f"Streaming not yet implemented for {llm_config.model_endpoint_type}")
        if not use_tool_naming:
            raise NotImplementedError("Only tool calling supported on Anthropic API requests")

        if functions is not None:
            tools = [{"type": "function", "function": f} for f in functions]
            tools = [Tool(**t) for t in tools]
        else:
            tools = None

        return anthropic_chat_completions_request(
            url=llm_config.model_endpoint,
            api_key=credentials.anthropic_key,
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

    elif llm_config.model_endpoint_type == "cohere":
        if stream:
            raise NotImplementedError(f"Streaming not yet implemented for {llm_config.model_endpoint_type}")
        if not use_tool_naming:
            raise NotImplementedError("Only tool calling supported on Cohere API requests")

        if functions is not None:
            tools = [{"type": "function", "function": f} for f in functions]
            tools = [Tool(**t) for t in tools]
        else:
            tools = None

        return cohere_chat_completions_request(
            # url=llm_config.model_endpoint,
            url="https://api.cohere.ai/v1",  # TODO
            api_key=os.getenv("COHERE_API_KEY"),  # TODO remove
            chat_completion_request=ChatCompletionRequest(
                model="command-r-plus",  # TODO
                messages=[cast_message_to_subtype(m.to_openai_dict()) for m in messages],
                tools=[{"type": "function", "function": f} for f in functions] if functions else None,
                tool_choice=function_call,
                # user=str(user_id),
                # NOTE: max_tokens is required for Anthropic API
                # max_tokens=1024,  # TODO make dynamic
            ),
        )

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
            auth_type=credentials.openllm_auth_type,
            auth_key=credentials.openllm_key,
        )
