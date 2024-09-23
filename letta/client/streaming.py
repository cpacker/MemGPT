import json
from typing import Generator

import httpx
from httpx_sse import SSEError, connect_sse

from letta.constants import OPENAI_CONTEXT_WINDOW_ERROR_SUBSTRING
from letta.errors import LLMError
from letta.schemas.enums import MessageStreamStatus
from letta.schemas.letta_message import (
    FunctionCallMessage,
    FunctionReturn,
    InternalMonologue,
)
from letta.schemas.letta_response import LettaStreamingResponse


def _sse_post(url: str, data: dict, headers: dict) -> Generator[LettaStreamingResponse, None, None]:

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
                    # if sse.data == OPENAI_SSE_DONE:
                    # print("finished")
                    # break
                    if sse.data in [status.value for status in MessageStreamStatus]:
                        # break
                        # print("sse.data::", sse.data)
                        yield MessageStreamStatus(sse.data)
                    else:
                        chunk_data = json.loads(sse.data)
                        if "internal_monologue" in chunk_data:
                            yield InternalMonologue(**chunk_data)
                        elif "function_call" in chunk_data:
                            yield FunctionCallMessage(**chunk_data)
                        elif "function_return" in chunk_data:
                            yield FunctionReturn(**chunk_data)
                        else:
                            raise ValueError(f"Unknown message type in chunk_data: {chunk_data}")

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
