import json
import warnings
from typing import Generator, List, Optional, Union

import httpx
import requests
from httpx_sse import connect_sse
from httpx_sse._exceptions import SSEError

from letta.constants import OPENAI_CONTEXT_WINDOW_ERROR_SUBSTRING
from letta.errors import LLMError
from letta.llm_api.helpers import (
    add_inner_thoughts_to_functions,
    convert_to_structured_output,
    make_post_request,
)
from letta.local_llm.constants import (
    INNER_THOUGHTS_KWARG,
    INNER_THOUGHTS_KWARG_DESCRIPTION,
)
from letta.local_llm.utils import num_tokens_from_functions, num_tokens_from_messages
from letta.schemas.llm_config import LLMConfig
from letta.schemas.message import Message as _Message
from letta.schemas.message import MessageRole as _MessageRole
from letta.schemas.openai.chat_completion_request import ChatCompletionRequest
from letta.schemas.openai.chat_completion_request import (
    FunctionCall as ToolFunctionChoiceFunctionCall,
)
from letta.schemas.openai.chat_completion_request import (
    Tool,
    ToolFunctionChoice,
    cast_message_to_subtype,
)
from letta.schemas.openai.chat_completion_response import (
    ChatCompletionChunkResponse,
    ChatCompletionResponse,
    Choice,
    FunctionCall,
    Message,
    ToolCall,
    UsageStatistics,
)
from letta.schemas.openai.embedding_response import EmbeddingResponse
from letta.streaming_interface import (
    AgentChunkStreamingInterface,
    AgentRefreshStreamingInterface,
)
from letta.utils import get_tool_call_id, smart_urljoin

OPENAI_SSE_DONE = "[DONE]"


def openai_get_model_list(
    url: str, api_key: Union[str, None], fix_url: Optional[bool] = False, extra_params: Optional[dict] = None
) -> dict:
    """https://platform.openai.com/docs/api-reference/models/list"""
    from letta.utils import printd

    # In some cases we may want to double-check the URL and do basic correction, eg:
    # In Letta config the address for vLLM is w/o a /v1 suffix for simplicity
    # However if we're treating the server as an OpenAI proxy we want the /v1 suffix on our model hit
    if fix_url:
        if not url.endswith("/v1"):
            url = smart_urljoin(url, "v1")

    url = smart_urljoin(url, "models")

    headers = {"Content-Type": "application/json"}
    if api_key is not None:
        headers["Authorization"] = f"Bearer {api_key}"

    printd(f"Sending request to {url}")
    response = None
    try:
        # TODO add query param "tool" to be true
        response = requests.get(url, headers=headers, params=extra_params)
        response.raise_for_status()  # Raises HTTPError for 4XX/5XX status
        response = response.json()  # convert to dict from string
        printd(f"response = {response}")
        return response
    except requests.exceptions.HTTPError as http_err:
        # Handle HTTP errors (e.g., response 4XX, 5XX)
        try:
            if response:
                response = response.json()
        except:
            pass
        printd(f"Got HTTPError, exception={http_err}, response={response}")
        raise http_err
    except requests.exceptions.RequestException as req_err:
        # Handle other requests-related errors (e.g., connection error)
        try:
            if response:
                response = response.json()
        except:
            pass
        printd(f"Got RequestException, exception={req_err}, response={response}")
        raise req_err
    except Exception as e:
        # Handle other potential errors
        try:
            if response:
                response = response.json()
        except:
            pass
        printd(f"Got unknown Exception, exception={e}, response={response}")
        raise e


def build_openai_chat_completions_request(
    llm_config: LLMConfig,
    messages: List[_Message],
    user_id: Optional[str],
    functions: Optional[list],
    function_call: Optional[str],
    use_tool_naming: bool,
    max_tokens: Optional[int],
) -> ChatCompletionRequest:
    if functions and llm_config.put_inner_thoughts_in_kwargs:
        functions = add_inner_thoughts_to_functions(
            functions=functions,
            inner_thoughts_key=INNER_THOUGHTS_KWARG,
            inner_thoughts_description=INNER_THOUGHTS_KWARG_DESCRIPTION,
        )

    openai_message_list = [
        cast_message_to_subtype(m.to_openai_dict(put_inner_thoughts_in_kwargs=llm_config.put_inner_thoughts_in_kwargs)) for m in messages
    ]
    if llm_config.model:
        model = llm_config.model
    else:
        warnings.warn(f"Model type not set in llm_config: {llm_config.model_dump_json(indent=4)}")
        model = None

    if use_tool_naming:
        if function_call is None:
            tool_choice = None
        elif function_call not in ["none", "auto", "required"]:
            tool_choice = ToolFunctionChoice(type="function", function=ToolFunctionChoiceFunctionCall(name=function_call))
        else:
            tool_choice = function_call
        data = ChatCompletionRequest(
            model=model,
            messages=openai_message_list,
            tools=[Tool(type="function", function=f) for f in functions] if functions else None,
            tool_choice=tool_choice,
            user=str(user_id),
            max_tokens=max_tokens,
        )
    else:
        data = ChatCompletionRequest(
            model=model,
            messages=openai_message_list,
            functions=functions,
            function_call=function_call,
            user=str(user_id),
            max_tokens=max_tokens,
        )
        # https://platform.openai.com/docs/guides/text-generation/json-mode
        # only supported by gpt-4o, gpt-4-turbo, or gpt-3.5-turbo
        # if "gpt-4o" in llm_config.model or "gpt-4-turbo" in llm_config.model or "gpt-3.5-turbo" in llm_config.model:
        # data.response_format = {"type": "json_object"}

    if "inference.memgpt.ai" in llm_config.model_endpoint:
        # override user id for inference.memgpt.ai
        import uuid

        data.user = str(uuid.UUID(int=0))
        data.model = "memgpt-openai"

    return data


def openai_chat_completions_process_stream(
    url: str,
    api_key: str,
    chat_completion_request: ChatCompletionRequest,
    stream_interface: Optional[Union[AgentChunkStreamingInterface, AgentRefreshStreamingInterface]] = None,
    create_message_id: bool = True,
    create_message_datetime: bool = True,
    override_tool_call_id: bool = True,
) -> ChatCompletionResponse:
    """Process a streaming completion response, and return a ChatCompletionRequest at the end.

    To "stream" the response in Letta, we want to call a streaming-compatible interface function
    on the chunks received from the OpenAI-compatible server POST SSE response.
    """
    assert chat_completion_request.stream == True
    assert stream_interface is not None, "Required"

    # Count the prompt tokens
    # TODO move to post-request?
    chat_history = [m.model_dump(exclude_none=True) for m in chat_completion_request.messages]
    # print(chat_history)

    prompt_tokens = num_tokens_from_messages(
        messages=chat_history,
        model=chat_completion_request.model,
    )
    # We also need to add the cost of including the functions list to the input prompt
    if chat_completion_request.tools is not None:
        assert chat_completion_request.functions is None
        prompt_tokens += num_tokens_from_functions(
            functions=[t.function.model_dump() for t in chat_completion_request.tools],
            model=chat_completion_request.model,
        )
    elif chat_completion_request.functions is not None:
        assert chat_completion_request.tools is None
        prompt_tokens += num_tokens_from_functions(
            functions=[f.model_dump() for f in chat_completion_request.functions],
            model=chat_completion_request.model,
        )

    # Create a dummy Message object to get an ID and date
    # TODO(sarah): add message ID generation function
    dummy_message = _Message(
        role=_MessageRole.assistant,
        text="",
        user_id="",
        agent_id="",
        model="",
        name=None,
        tool_calls=None,
        tool_call_id=None,
    )

    TEMP_STREAM_RESPONSE_ID = "temp_id"
    TEMP_STREAM_FINISH_REASON = "temp_null"
    TEMP_STREAM_TOOL_CALL_ID = "temp_id"
    chat_completion_response = ChatCompletionResponse(
        id=dummy_message.id if create_message_id else TEMP_STREAM_RESPONSE_ID,
        choices=[],
        created=dummy_message.created_at,  # NOTE: doesn't matter since both will do get_utc_time()
        model=chat_completion_request.model,
        usage=UsageStatistics(
            completion_tokens=0,
            prompt_tokens=prompt_tokens,
            total_tokens=prompt_tokens,
        ),
    )

    if stream_interface:
        stream_interface.stream_start()

    n_chunks = 0  # approx == n_tokens
    try:
        for chunk_idx, chat_completion_chunk in enumerate(
            openai_chat_completions_request_stream(url=url, api_key=api_key, chat_completion_request=chat_completion_request)
        ):
            assert isinstance(chat_completion_chunk, ChatCompletionChunkResponse), type(chat_completion_chunk)

            # NOTE: this assumes that the tool call ID will only appear in one of the chunks during the stream
            if override_tool_call_id:
                for choice in chat_completion_chunk.choices:
                    if choice.delta.tool_calls and len(choice.delta.tool_calls) > 0:
                        for tool_call in choice.delta.tool_calls:
                            if tool_call.id is not None:
                                tool_call.id = get_tool_call_id()

            if stream_interface:
                if isinstance(stream_interface, AgentChunkStreamingInterface):
                    stream_interface.process_chunk(
                        chat_completion_chunk,
                        message_id=chat_completion_response.id if create_message_id else chat_completion_chunk.id,
                        message_date=chat_completion_response.created if create_message_datetime else chat_completion_chunk.created,
                    )
                elif isinstance(stream_interface, AgentRefreshStreamingInterface):
                    stream_interface.process_refresh(chat_completion_response)
                else:
                    raise TypeError(stream_interface)

            if chunk_idx == 0:
                # initialize the choice objects which we will increment with the deltas
                num_choices = len(chat_completion_chunk.choices)
                assert num_choices > 0
                chat_completion_response.choices = [
                    Choice(
                        finish_reason=TEMP_STREAM_FINISH_REASON,  # NOTE: needs to be ovrerwritten
                        index=i,
                        message=Message(
                            role="assistant",
                        ),
                    )
                    for i in range(len(chat_completion_chunk.choices))
                ]

            # add the choice delta
            assert len(chat_completion_chunk.choices) == len(chat_completion_response.choices), chat_completion_chunk
            for chunk_choice in chat_completion_chunk.choices:
                if chunk_choice.finish_reason is not None:
                    chat_completion_response.choices[chunk_choice.index].finish_reason = chunk_choice.finish_reason

                if chunk_choice.logprobs is not None:
                    chat_completion_response.choices[chunk_choice.index].logprobs = chunk_choice.logprobs

                accum_message = chat_completion_response.choices[chunk_choice.index].message
                message_delta = chunk_choice.delta

                if message_delta.content is not None:
                    content_delta = message_delta.content
                    if accum_message.content is None:
                        accum_message.content = content_delta
                    else:
                        accum_message.content += content_delta

                # TODO(charles) make sure this works for parallel tool calling?
                if message_delta.tool_calls is not None:
                    tool_calls_delta = message_delta.tool_calls

                    # If this is the first tool call showing up in a chunk, initialize the list with it
                    if accum_message.tool_calls is None:
                        accum_message.tool_calls = [
                            ToolCall(id=TEMP_STREAM_TOOL_CALL_ID, function=FunctionCall(name="", arguments=""))
                            for _ in range(len(tool_calls_delta))
                        ]

                    # There may be many tool calls in a tool calls delta (e.g. parallel tool calls)
                    for tool_call_delta in tool_calls_delta:
                        if tool_call_delta.id is not None:
                            # TODO assert that we're not overwriting?
                            # TODO += instead of =?
                            if tool_call_delta.index not in range(len(accum_message.tool_calls)):
                                warnings.warn(
                                    f"Tool call index out of range ({tool_call_delta.index})\ncurrent tool calls: {accum_message.tool_calls}\ncurrent delta: {tool_call_delta}"
                                )
                            else:
                                accum_message.tool_calls[tool_call_delta.index].id = tool_call_delta.id
                        if tool_call_delta.function is not None:
                            if tool_call_delta.function.name is not None:
                                # TODO assert that we're not overwriting?
                                # TODO += instead of =?
                                accum_message.tool_calls[tool_call_delta.index].function.name = tool_call_delta.function.name
                            if tool_call_delta.function.arguments is not None:
                                accum_message.tool_calls[tool_call_delta.index].function.arguments += tool_call_delta.function.arguments

                if message_delta.function_call is not None:
                    raise NotImplementedError(f"Old function_call style not support with stream=True")

            # overwrite response fields based on latest chunk
            if not create_message_id:
                chat_completion_response.id = chat_completion_chunk.id
            if not create_message_datetime:
                chat_completion_response.created = chat_completion_chunk.created
            chat_completion_response.model = chat_completion_chunk.model
            chat_completion_response.system_fingerprint = chat_completion_chunk.system_fingerprint

            # increment chunk counter
            n_chunks += 1

    except Exception as e:
        if stream_interface:
            stream_interface.stream_end()
        print(f"Parsing ChatCompletion stream failed with error:\n{str(e)}")
        raise e
    finally:
        if stream_interface:
            stream_interface.stream_end()

    # make sure we didn't leave temp stuff in
    assert all([c.finish_reason != TEMP_STREAM_FINISH_REASON for c in chat_completion_response.choices])
    assert all(
        [
            all([tc.id != TEMP_STREAM_TOOL_CALL_ID for tc in c.message.tool_calls]) if c.message.tool_calls else True
            for c in chat_completion_response.choices
        ]
    )
    if not create_message_id:
        assert chat_completion_response.id != dummy_message.id

    # compute token usage before returning
    # TODO try actually computing the #tokens instead of assuming the chunks is the same
    chat_completion_response.usage.completion_tokens = n_chunks
    chat_completion_response.usage.total_tokens = prompt_tokens + n_chunks

    assert len(chat_completion_response.choices) > 0, chat_completion_response

    # printd(chat_completion_response)
    return chat_completion_response


def _sse_post(url: str, data: dict, headers: dict) -> Generator[ChatCompletionChunkResponse, None, None]:

    with httpx.Client() as client:
        with connect_sse(client, method="POST", url=url, json=data, headers=headers) as event_source:

            # Inspect for errors before iterating (see https://github.com/florimondmanca/httpx-sse/pull/12)
            if not event_source.response.is_success:
                # handle errors
                from letta.utils import printd

                printd("Caught error before iterating SSE request:", vars(event_source.response))
                printd(event_source.response.read())

                try:
                    response_bytes = event_source.response.read()
                    response_dict = json.loads(response_bytes.decode("utf-8"))
                    error_message = response_dict["error"]["message"]
                    # e.g.: This model's maximum context length is 8192 tokens. However, your messages resulted in 8198 tokens (7450 in the messages, 748 in the functions). Please reduce the length of the messages or functions.
                    if OPENAI_CONTEXT_WINDOW_ERROR_SUBSTRING in error_message:
                        raise LLMError(error_message)
                except LLMError:
                    raise
                except:
                    print(f"Failed to parse SSE message, throwing SSE HTTP error up the stack")
                    event_source.response.raise_for_status()

            try:
                for sse in event_source.iter_sse():
                    # printd(sse.event, sse.data, sse.id, sse.retry)
                    if sse.data == OPENAI_SSE_DONE:
                        # print("finished")
                        break
                    else:
                        chunk_data = json.loads(sse.data)
                        # print("chunk_data::", chunk_data)
                        chunk_object = ChatCompletionChunkResponse(**chunk_data)
                        # print("chunk_object::", chunk_object)
                        # id=chunk_data["id"],
                        # choices=[ChunkChoice],
                        # model=chunk_data["model"],
                        # system_fingerprint=chunk_data["system_fingerprint"]
                        # )
                        yield chunk_object

            except SSEError as e:
                print("Caught an error while iterating the SSE stream:", str(e))
                if "application/json" in str(e):  # Check if the error is because of JSON response
                    # TODO figure out a better way to catch the error other than re-trying with a POST
                    response = client.post(url=url, json=data, headers=headers)  # Make the request again to get the JSON response
                    if response.headers["Content-Type"].startswith("application/json"):
                        error_details = response.json()  # Parse the JSON to get the error message
                        print("Request:", vars(response.request))
                        print("POST Error:", error_details)
                        print("Original SSE Error:", str(e))
                    else:
                        print("Failed to retrieve JSON error message via retry.")
                else:
                    print("SSEError not related to 'application/json' content type.")

                # Optionally re-raise the exception if you need to propagate it
                raise e

            except Exception as e:
                if event_source.response.request is not None:
                    print("HTTP Request:", vars(event_source.response.request))
                if event_source.response is not None:
                    print("HTTP Status:", event_source.response.status_code)
                    print("HTTP Headers:", event_source.response.headers)
                    # print("HTTP Body:", event_source.response.text)
                print("Exception message:", str(e))
                raise e


def openai_chat_completions_request_stream(
    url: str,
    api_key: str,
    chat_completion_request: ChatCompletionRequest,
) -> Generator[ChatCompletionChunkResponse, None, None]:
    from letta.utils import printd

    url = smart_urljoin(url, "chat/completions")
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    data = chat_completion_request.model_dump(exclude_none=True)

    printd("Request:\n", json.dumps(data, indent=2))

    # If functions == None, strip from the payload
    if "functions" in data and data["functions"] is None:
        data.pop("functions")
        data.pop("function_call", None)  # extra safe,  should exist always (default="auto")

    if "tools" in data and data["tools"] is None:
        data.pop("tools")
        data.pop("tool_choice", None)  # extra safe,  should exist always (default="auto")

    if "tools" in data:
        for tool in data["tools"]:
            # tool["strict"] = True
            tool["function"] = convert_to_structured_output(tool["function"])

    # print(f"\n\n\n\nData[tools]: {json.dumps(data['tools'], indent=2)}")

    printd(f"Sending request to {url}")
    try:
        return _sse_post(url=url, data=data, headers=headers)
    except requests.exceptions.HTTPError as http_err:
        # Handle HTTP errors (e.g., response 4XX, 5XX)
        printd(f"Got HTTPError, exception={http_err}, payload={data}")
        raise http_err
    except requests.exceptions.RequestException as req_err:
        # Handle other requests-related errors (e.g., connection error)
        printd(f"Got RequestException, exception={req_err}")
        raise req_err
    except Exception as e:
        # Handle other potential errors
        printd(f"Got unknown Exception, exception={e}")
        raise e


def openai_chat_completions_request(
    url: str,
    api_key: str,
    chat_completion_request: ChatCompletionRequest,
) -> ChatCompletionResponse:
    """Send a ChatCompletion request to an OpenAI-compatible server

    If request.stream == True, will yield ChatCompletionChunkResponses
    If request.stream == False, will return a ChatCompletionResponse

    https://platform.openai.com/docs/guides/text-generation?lang=curl
    """
    from letta.utils import printd

    url = smart_urljoin(url, "chat/completions")
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    data = chat_completion_request.model_dump(exclude_none=True)

    # add check otherwise will cause error: "Invalid value for 'parallel_tool_calls': 'parallel_tool_calls' is only allowed when 'tools' are specified."
    if chat_completion_request.tools is not None:
        data["parallel_tool_calls"] = False

    printd("Request:\n", json.dumps(data, indent=2))

    # If functions == None, strip from the payload
    if "functions" in data and data["functions"] is None:
        data.pop("functions")
        data.pop("function_call", None)  # extra safe,  should exist always (default="auto")

    if "tools" in data and data["tools"] is None:
        data.pop("tools")
        data.pop("tool_choice", None)  # extra safe,  should exist always (default="auto")

    response_json = make_post_request(url, headers, data)
    return ChatCompletionResponse(**response_json)


def openai_embeddings_request(url: str, api_key: str, data: dict) -> EmbeddingResponse:
    """https://platform.openai.com/docs/api-reference/embeddings/create"""

    url = smart_urljoin(url, "embeddings")
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    response_json = make_post_request(url, headers, data)
    return EmbeddingResponse(**response_json)
