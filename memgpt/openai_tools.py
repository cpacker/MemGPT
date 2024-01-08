import random
import time
import requests
import time
from typing import Callable, TypeVar, Union
import urllib

from box import Box

from memgpt.local_llm.chat_completion_proxy import get_chat_completion
from memgpt.constants import CLI_WARNING_PREFIX

MODEL_TO_AZURE_ENGINE = {
    "gpt-4-1106-preview": "gpt-4",
    "gpt-4": "gpt-4",
    "gpt-4-32k": "gpt-4-32k",
    "gpt-3.5": "gpt-35-turbo",
    "gpt-3.5-turbo": "gpt-35-turbo",
    "gpt-3.5-turbo-16k": "gpt-35-turbo-16k",
}


def is_context_overflow_error(exception):
    from memgpt.utils import printd

    match_string = "maximum context length"

    # Backwards compatability with openai python package/client v0.28 (pre-v1 client migration)
    if match_string in str(exception):
        printd(f"Found '{match_string}' in str(exception)={(str(exception))}")
        return True

    # Based on python requests + OpenAI REST API (/v1)
    elif isinstance(exception, requests.exceptions.HTTPError):
        if exception.response is not None and "application/json" in exception.response.headers.get("Content-Type", ""):
            try:
                error_details = exception.response.json()
                if "error" not in error_details:
                    printd(f"HTTPError occured, but couldn't find error field: {error_details}")
                    return False
                else:
                    error_details = error_details["error"]

                # Check for the specific error code
                if error_details.get("code") == "context_length_exceeded":
                    printd(f"HTTPError occured, caught error code {error_details.get('code')}")
                    return True
                # Soft-check for "maximum context length" inside of the message
                elif error_details.get("message") and "maximum context length" in error_details.get("message"):
                    printd(f"HTTPError occured, found '{match_string}' in error message contents ({error_details})")
                    return True
                else:
                    printd(f"HTTPError occured, but unknown error message: {error_details}")
                    return False
            except ValueError:
                # JSON decoding failed
                printd(f"HTTPError occurred ({exception}), but no JSON error message.")

    # Generic fail
    else:
        return False


def smart_urljoin(base_url, relative_url):
    """urljoin is stupid and wants a trailing / at the end of the endpoint address, or it will chop the suffix off"""
    if not base_url.endswith("/"):
        base_url += "/"
    return urllib.parse.urljoin(base_url, relative_url)


def clean_azure_endpoint(raw_endpoint_name):
    """Make sure the endpoint is of format 'https://YOUR_RESOURCE_NAME.openai.azure.com'"""
    endpoint_address = raw_endpoint_name.strip("/").replace(".openai.azure.com", "")
    endpoint_address = endpoint_address.replace("http://", "")
    endpoint_address = endpoint_address.replace("https://", "")
    return endpoint_address


def openai_get_model_list(url: str, api_key: Union[str, None]) -> dict:
    """https://platform.openai.com/docs/api-reference/models/list"""
    from memgpt.utils import printd

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


def azure_openai_get_model_list(url: str, api_key: Union[str, None], api_version: str) -> dict:
    """https://learn.microsoft.com/en-us/rest/api/azureopenai/models/list?view=rest-azureopenai-2023-05-15&tabs=HTTP"""
    from memgpt.utils import printd

    # https://xxx.openai.azure.com/openai/models?api-version=xxx
    url = smart_urljoin(url, "openai")
    url = smart_urljoin(url, f"models?api-version={api_version}")

    headers = {"Content-Type": "application/json"}
    if api_key is not None:
        headers["api-key"] = f"{api_key}"

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


def openai_chat_completions_request(url, api_key, data):
    """https://platform.openai.com/docs/guides/text-generation?lang=curl"""
    from memgpt.utils import printd

    url = smart_urljoin(url, "chat/completions")
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}

    # If functions == None, strip from the payload
    if "functions" in data and data["functions"] is None:
        data.pop("functions")
        data.pop("function_call", None)  # extra safe,  should exist always (default="auto")

    printd(f"Sending request to {url}")
    try:
        # Example code to trigger a rate limit response:
        # mock_response = requests.Response()
        # mock_response.status_code = 429
        # http_error = requests.exceptions.HTTPError("429 Client Error: Too Many Requests")
        # http_error.response = mock_response
        # raise http_error

        # Example code to trigger a context overflow response (for an 8k model)
        # data["messages"][-1]["content"] = " ".join(["repeat after me this is not a fluke"] * 1000)

        response = requests.post(url, headers=headers, json=data)
        printd(f"response = {response}")
        response.raise_for_status()  # Raises HTTPError for 4XX/5XX status
        response = response.json()  # convert to dict from string
        printd(f"response.json = {response}")
        response = Box(response)  # convert to 'dot-dict' style which is the openai python client default
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


def openai_embeddings_request(url, api_key, data):
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
        response = Box(response)  # convert to 'dot-dict' style which is the openai python client default
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


def azure_openai_chat_completions_request(resource_name, deployment_id, api_version, api_key, data):
    """https://learn.microsoft.com/en-us/azure/ai-services/openai/reference#chat-completions"""
    from memgpt.utils import printd

    resource_name = clean_azure_endpoint(resource_name)
    url = f"https://{resource_name}.openai.azure.com/openai/deployments/{deployment_id}/chat/completions?api-version={api_version}"
    headers = {"Content-Type": "application/json", "api-key": f"{api_key}"}

    # If functions == None, strip from the payload
    if "functions" in data and data["functions"] is None:
        data.pop("functions")
        data.pop("function_call", None)  # extra safe,  should exist always (default="auto")

    printd(f"Sending request to {url}")
    try:
        response = requests.post(url, headers=headers, json=data)
        printd(f"response = {response}")
        response.raise_for_status()  # Raises HTTPError for 4XX/5XX status
        response = response.json()  # convert to dict from string
        printd(f"response.json = {response}")
        # NOTE: azure openai does not include "content" in the response when it is None, so we need to add it
        if "content" not in response["choices"][0].get("message"):
            response["choices"][0]["message"]["content"] = None
        response = Box(response)  # convert to 'dot-dict' style which is the openai python client default
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


def azure_openai_embeddings_request(resource_name, deployment_id, api_version, api_key, data):
    """https://learn.microsoft.com/en-us/azure/ai-services/openai/reference#embeddings"""
    from memgpt.utils import printd

    resource_name = clean_azure_endpoint(resource_name)
    url = f"https://{resource_name}.openai.azure.com/openai/deployments/{deployment_id}/embeddings?api-version={api_version}"
    headers = {"Content-Type": "application/json", "api-key": f"{api_key}"}

    printd(f"Sending request to {url}")
    try:
        response = requests.post(url, headers=headers, json=data)
        printd(f"response = {response}")
        response.raise_for_status()  # Raises HTTPError for 4XX/5XX status
        response = response.json()  # convert to dict from string
        printd(f"response.json = {response}")
        response = Box(response)  # convert to 'dot-dict' style which is the openai python client default
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
        from memgpt.utils import printd

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
    agent_config,
    messages,
    functions=None,
    function_call="auto",
    # hint
    first_message=False,
):
    """Return response to chat completion with backoff"""
    from memgpt.utils import printd
    from memgpt.config import MemGPTConfig

    config = MemGPTConfig.load()  # load credentials (currently not stored in agent config)

    printd(f"Using model {agent_config.llm_config.model_endpoint_type}, endpoint: {agent_config.llm_config.model_endpoint}")
    if agent_config.llm_config.model_endpoint_type == "openai":
        # openai
        return openai_chat_completions_request(
            url=agent_config.llm_config.model_endpoint,  # https://api.openai.com/v1 -> https://api.openai.com/v1/chat/completions
            api_key=config.openai_key,  # 'sk....'
            data=dict(
                model=agent_config.llm_config.model,
                messages=messages,
                functions=functions,
                function_call=function_call,
                user=config.anon_clientid,
            ),
        )
    elif agent_config.llm_config.model_endpoint_type == "azure":
        # azure
        azure_deployment = (
            config.azure_deployment if config.azure_deployment is not None else MODEL_TO_AZURE_ENGINE[agent_config.llm_config.model]
        )
        return azure_openai_chat_completions_request(
            resource_name=config.azure_endpoint,
            deployment_id=azure_deployment,
            api_version=config.azure_version,
            api_key=config.azure_key,
            data=dict(
                # NOTE: don't pass model to Azure calls, that is the deployment_id
                # model=agent_config.model,
                messages=messages,
                functions=functions,
                function_call=function_call,
                user=config.anon_clientid,
            ),
        )
    else:  # local model
        return get_chat_completion(
            model=agent_config.llm_config.model,
            messages=messages,
            functions=functions,
            function_call=function_call,
            context_window=agent_config.llm_config.context_window,
            endpoint=agent_config.llm_config.model_endpoint,
            endpoint_type=agent_config.llm_config.model_endpoint_type,
            wrapper=agent_config.llm_config.model_wrapper,
            user=config.anon_clientid,
            # hint
            first_message=first_message,
        )
