import json
from typing import Generator, Optional, Union

import httpx
import requests
from httpx_sse import connect_sse
from httpx_sse._exceptions import SSEError

from memgpt.local_llm.utils import num_tokens_from_functions, num_tokens_from_messages
from memgpt.models.chat_completion_request import ChatCompletionRequest
from memgpt.models.chat_completion_response import (
    ChatCompletionChunkResponse,
    ChatCompletionResponse,
    Choice,
    FunctionCall,
    Message,
    ToolCall,
    UsageStatistics,
)
from memgpt.models.embedding_response import EmbeddingResponse
from memgpt.streaming_interface import (
    AgentChunkStreamingInterface,
    AgentRefreshStreamingInterface,
)
from memgpt.utils import get_utc_time, smart_urljoin

OPENAI_SSE_DONE = "[DONE]"


def openai_get_model_list(url: str, api_key: Union[str, None], fix_url: Optional[bool] = False) -> dict:
    """https://platform.openai.com/docs/api-reference/models/list"""
    from memgpt.utils import printd

    # In some cases we may want to double-check the URL and do basic correction, eg:
    # In MemGPT config the address for vLLM is w/o a /v1 suffix for simplicity
    # However if we're treating the server as an OpenAI proxy we want the /v1 suffix on our model hit
    if fix_url:
        if not url.endswith("/v1"):
            url = smart_urljoin(url, "v1")

    url = smart_urljoin(url, "models")

    headers = {"Content-Type": "application/json"}
    if api_key is not None:
        headers["Authorization"] = f"Bearer {api_key}"

    printd(f"Sending request to {url}")
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Raises HTTPError for 4XX/5XX status
        response = response.json()  # convert to dict from string
        printd(f"response = {response}")
        return response
    except requests.exceptions.HTTPError as http_err:
        # Handle HTTP errors (e.g., response 4XX, 5XX)
        try:
            response = response.json()
        except:
            pass
        printd(f"Got HTTPError, exception={http_err}, response={response}")
        raise http_err
    except requests.exceptions.RequestException as req_err:
        # Handle other requests-related errors (e.g., connection error)
        try:
            response = response.json()
        except:
            pass
        printd(f"Got RequestException, exception={req_err}, response={response}")
        raise req_err
    except Exception as e:
        # Handle other potential errors
        try:
            response = response.json()
        except:
            pass
        printd(f"Got unknown Exception, exception={e}, response={response}")
        raise e


def openai_chat_completions_process_stream(
    url: str,
    api_key: str,
    chat_completion_request: ChatCompletionRequest,
    stream_inferface: Optional[Union[AgentChunkStreamingInterface, AgentRefreshStreamingInterface]] = None,
) -> ChatCompletionResponse:
    """Process a streaming completion response, and return a ChatCompletionRequest at the end.

    To "stream" the response in MemGPT, we want to call a streaming-compatible interface function
    on the chunks received from the OpenAI-compatible server POST SSE response.
    """
    assert chat_completion_request.stream == True

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

    TEMP_STREAM_RESPONSE_ID = "temp_id"
    TEMP_STREAM_FINISH_REASON = "temp_null"
    TEMP_STREAM_TOOL_CALL_ID = "temp_id"
    chat_completion_response = ChatCompletionResponse(
        id=TEMP_STREAM_RESPONSE_ID,
        choices=[],
        created=get_utc_time(),
        model=chat_completion_request.model,
        usage=UsageStatistics(
            completion_tokens=0,
            prompt_tokens=prompt_tokens,
            total_tokens=prompt_tokens,
        ),
    )

    if stream_inferface:
        stream_inferface.stream_start()

    n_chunks = 0  # approx == n_tokens
    try:
        for chunk_idx, chat_completion_chunk in enumerate(
            openai_chat_completions_request_stream(url=url, api_key=api_key, chat_completion_request=chat_completion_request)
        ):
            assert isinstance(chat_completion_chunk, ChatCompletionChunkResponse), type(chat_completion_chunk)
            # print(chat_completion_chunk)

            if stream_inferface:
                if isinstance(stream_inferface, AgentChunkStreamingInterface):
                    stream_inferface.process_chunk(chat_completion_chunk)
                elif isinstance(stream_inferface, AgentRefreshStreamingInterface):
                    stream_inferface.process_refresh(chat_completion_response)
                else:
                    raise TypeError(stream_inferface)

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

                if message_delta.tool_calls is not None:
                    tool_calls_delta = message_delta.tool_calls

                    # If this is the first tool call showing up in a chunk, initialize the list with it
                    if accum_message.tool_calls is None:
                        accum_message.tool_calls = [
                            ToolCall(id=TEMP_STREAM_TOOL_CALL_ID, function=FunctionCall(name="", arguments=""))
                            for _ in range(len(tool_calls_delta))
                        ]

                    for tool_call_delta in tool_calls_delta:
                        if tool_call_delta.id is not None:
                            # TODO assert that we're not overwriting?
                            # TODO += instead of =?
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
            chat_completion_response.id = chat_completion_chunk.id
            chat_completion_response.system_fingerprint = chat_completion_chunk.system_fingerprint
            chat_completion_response.created = chat_completion_chunk.created
            chat_completion_response.model = chat_completion_chunk.model

            # increment chunk counter
            n_chunks += 1

    except Exception as e:
        if stream_inferface:
            stream_inferface.stream_end()
        print(f"Parsing ChatCompletion stream failed with error:\n{str(e)}")
        raise e
    finally:
        if stream_inferface:
            stream_inferface.stream_end()

    # make sure we didn't leave temp stuff in
    assert all([c.finish_reason != TEMP_STREAM_FINISH_REASON for c in chat_completion_response.choices])
    assert all(
        [
            all([tc != TEMP_STREAM_TOOL_CALL_ID for tc in c.message.tool_calls]) if c.message.tool_calls else True
            for c in chat_completion_response.choices
        ]
    )
    assert chat_completion_response.id != TEMP_STREAM_RESPONSE_ID

    # compute token usage before returning
    # TODO try actually computing the #tokens instead of assuming the chunks is the same
    chat_completion_response.usage.completion_tokens = n_chunks
    chat_completion_response.usage.total_tokens = prompt_tokens + n_chunks

    # printd(chat_completion_response)
    return chat_completion_response


def _sse_post(url: str, data: dict, headers: dict) -> Generator[ChatCompletionChunkResponse, None, None]:

    with httpx.Client() as client:
        with connect_sse(client, method="POST", url=url, json=data, headers=headers) as event_source:
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
                if "application/json" in str(e):  # Check if the error is because of JSON response
                    response = client.post(url=url, json=data, headers=headers)  # Make the request again to get the JSON response
                    if response.headers["Content-Type"].startswith("application/json"):
                        error_details = response.json()  # Parse the JSON to get the error message
                        print("Error:", error_details)
                        print("Reqeust:", vars(response.request))
                    else:
                        print("Failed to retrieve JSON error message.")
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
    from memgpt.utils import printd

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
    from memgpt.utils import printd

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

    printd(f"Sending request to {url}")
    try:
        response = requests.post(url, headers=headers, json=data)
        # printd(f"response = {response}, response.text = {response.text}")
        response.raise_for_status()  # Raises HTTPError for 4XX/5XX status

        response = response.json()  # convert to dict from string
        printd(f"response.json = {response}")

        response = ChatCompletionResponse(**response)  # convert to 'dot-dict' style which is the openai python client default
        return response
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


def openai_embeddings_request(url: str, api_key: str, data: dict) -> EmbeddingResponse:
    """https://platform.openai.com/docs/api-reference/embeddings/create"""
    from memgpt.utils import printd

    url = smart_urljoin(url, "embeddings")
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}

    printd(f"Sending request to {url}")
    try:
        response = requests.post(url, headers=headers, json=data)
        printd(f"response = {response}")
        response.raise_for_status()  # Raises HTTPError for 4XX/5XX status
        response = response.json()  # convert to dict from string
        printd(f"response.json = {response}")
        response = EmbeddingResponse(**response)  # convert to 'dot-dict' style which is the openai python client default
        return response
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
