import copy
import json
import warnings
from typing import Any, List, Union

import requests

from letta.constants import OPENAI_CONTEXT_WINDOW_ERROR_SUBSTRING
from letta.schemas.enums import OptionState
from letta.schemas.openai.chat_completion_response import ChatCompletionResponse, Choice
from letta.utils import json_dumps, printd


def make_post_request(url: str, headers: dict[str, str], data: dict[str, Any]) -> dict[str, Any]:
    printd(f"Sending request to {url}")
    try:
        # Make the POST request
        response = requests.post(url, headers=headers, json=data)
        printd(f"Response status code: {response.status_code}")

        # Raise for 4XX/5XX HTTP errors
        response.raise_for_status()

        # Check if the response content type indicates JSON and attempt to parse it
        content_type = response.headers.get("Content-Type", "")
        if "application/json" in content_type.lower():
            try:
                response_data = response.json()  # Attempt to parse the response as JSON
                printd(f"Response JSON: {response_data}")
            except ValueError as json_err:
                # Handle the case where the content type says JSON but the body is invalid
                error_message = f"Failed to parse JSON despite Content-Type being {content_type}: {json_err}"
                printd(error_message)
                raise ValueError(error_message) from json_err
        else:
            error_message = f"Unexpected content type returned: {response.headers.get('Content-Type')}"
            printd(error_message)
            raise ValueError(error_message)

        # Process the response using the callback function
        return response_data

    except requests.exceptions.HTTPError as http_err:
        # HTTP errors (4XX, 5XX)
        error_message = f"HTTP error occurred: {http_err}"
        if http_err.response is not None:
            error_message += f" | Status code: {http_err.response.status_code}, Message: {http_err.response.text}"
        printd(error_message)
        raise requests.exceptions.HTTPError(error_message) from http_err

    except requests.exceptions.Timeout as timeout_err:
        # Handle timeout errors
        error_message = f"Request timed out: {timeout_err}"
        printd(error_message)
        raise requests.exceptions.Timeout(error_message) from timeout_err

    except requests.exceptions.RequestException as req_err:
        # Non-HTTP errors (e.g., connection, SSL errors)
        error_message = f"Request failed: {req_err}"
        printd(error_message)
        raise requests.exceptions.RequestException(error_message) from req_err

    except ValueError as val_err:
        # Handle content-type or non-JSON response issues
        error_message = f"ValueError: {val_err}"
        printd(error_message)
        raise ValueError(error_message) from val_err

    except Exception as e:
        # Catch any other unknown exceptions
        error_message = f"An unexpected error occurred: {e}"
        printd(error_message)
        raise Exception(error_message) from e


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


def unpack_all_inner_thoughts_from_kwargs(
    response: ChatCompletionResponse,
    inner_thoughts_key: str,
) -> ChatCompletionResponse:
    """Strip the inner thoughts out of the tool call and put it in the message content"""
    if len(response.choices) == 0:
        raise ValueError(f"Unpacking inner thoughts from empty response not supported")

    new_choices = []
    for choice in response.choices:
        new_choices.append(unpack_inner_thoughts_from_kwargs(choice, inner_thoughts_key))

    # return an updated copy
    new_response = response.model_copy(deep=True)
    new_response.choices = new_choices
    return new_response


def unpack_inner_thoughts_from_kwargs(choice: Choice, inner_thoughts_key: str) -> Choice:
    message = choice.message
    if message.role == "assistant" and message.tool_calls and len(message.tool_calls) >= 1:
        if len(message.tool_calls) > 1:
            warnings.warn(f"Unpacking inner thoughts from more than one tool call ({len(message.tool_calls)}) is not supported")
        # TODO support multiple tool calls
        tool_call = message.tool_calls[0]

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

                return new_choice
            else:
                warnings.warn(f"Did not find inner thoughts in tool call: {str(tool_call)}")
                return choice

        except json.JSONDecodeError as e:
            warnings.warn(f"Failed to strip inner thoughts from kwargs: {e}")
            raise e


def is_context_overflow_error(exception: Union[requests.exceptions.RequestException, Exception]) -> bool:
    """Checks if an exception is due to context overflow (based on common OpenAI response messages)"""
    from letta.utils import printd

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


def derive_inner_thoughts_in_kwargs(inner_thoughts_in_kwargs_option: OptionState, model: str):
    if inner_thoughts_in_kwargs_option == OptionState.DEFAULT:
        # model that are known to not use `content` fields on tool calls
        inner_thoughts_in_kwargs = "gpt-4o" in model or "gpt-4-turbo" in model or "gpt-3.5-turbo" in model
    else:
        inner_thoughts_in_kwargs = True if inner_thoughts_in_kwargs_option == OptionState.YES else False

    if not isinstance(inner_thoughts_in_kwargs, bool):
        warnings.warn(f"Bad type detected: {type(inner_thoughts_in_kwargs)}")
        inner_thoughts_in_kwargs = bool(inner_thoughts_in_kwargs)

    return inner_thoughts_in_kwargs
