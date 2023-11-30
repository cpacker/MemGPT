import random
import os
import time
import requests
import time
from typing import Callable, TypeVar
import urllib

from box import Box

from memgpt.local_llm.chat_completion_proxy import get_chat_completion

HOST = os.getenv("OPENAI_API_BASE")
HOST_TYPE = os.getenv("BACKEND_TYPE")  # default None == ChatCompletion
R = TypeVar("R")


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


def openai_chat_completions_request(url, api_key, data):
    """https://platform.openai.com/docs/guides/text-generation?lang=curl"""
    from memgpt.utils import printd

    url = smart_urljoin(url, "chat/completions")
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}

    printd(f"Sending request to {url}")
    try:
        # from unittest.mock import MagicMock
        # mock_response = requests.Response()
        # mock_response.status_code = 429
        # http_error = requests.exceptions.HTTPError("429 Client Error: Too Many Requests")
        # http_error.response = mock_response
        # raise http_error

        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()  # Raises HTTPError for 4XX/5XX status
        response = response.json()  # convert to dict from string
        response = Box(response)  # convert to 'dot-dict' style which is the openai python client default
        return response
    except requests.exceptions.HTTPError as http_err:
        # Handle HTTP errors (e.g., response 4XX, 5XX)
        raise http_err
    except requests.exceptions.RequestException as req_err:
        # Handle other requests-related errors (e.g., connection error)
        raise req_err
    except Exception as e:
        # Handle other potential errors
        raise e


def openai_embeddings_request(url, api_key, data):
    """https://platform.openai.com/docs/api-reference/embeddings/create"""
    from memgpt.utils import printd

    url = smart_urljoin(url, "embeddings")
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}

    printd(f"Sending request to {url}")
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()  # Raises HTTPError for 4XX/5XX status
        response = response.json()  # convert to dict from string
        response = Box(response)  # convert to 'dot-dict' style which is the openai python client default
        return response
    except requests.exceptions.HTTPError as http_err:
        # Handle HTTP errors (e.g., response 4XX, 5XX)
        raise http_err
    except requests.exceptions.RequestException as req_err:
        # Handle other requests-related errors (e.g., connection error)
        raise req_err
    except Exception as e:
        # Handle other potential errors
        raise e


def azure_openai_chat_completions_request(resource_name, deployment_id, api_version, api_key, data):
    """https://learn.microsoft.com/en-us/azure/ai-services/openai/reference#chat-completions"""
    from memgpt.utils import printd

    resource_name = clean_azure_endpoint(resource_name)
    url = f"https://{resource_name}.openai.azure.com/openai/deployments/{deployment_id}/chat/completions?api-version={api_version}"
    headers = {"Content-Type": "application/json", "api-key": f"{api_key}"}

    printd(f"Sending request to {url}")
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()  # Raises HTTPError for 4XX/5XX status
        response = response.json()  # convert to dict from string
        # NOTE: azure openai does not include "content" in the response when it is None, so we need to add it
        if "content" not in response["choices"][0].get("message"):
            response["choices"][0]["message"]["content"] = None
        response = Box(response)  # convert to 'dot-dict' style which is the openai python client default
        return response
    except requests.exceptions.HTTPError as http_err:
        # Handle HTTP errors (e.g., response 4XX, 5XX)
        raise http_err
    except requests.exceptions.RequestException as req_err:
        # Handle other requests-related errors (e.g., connection error)
        raise req_err
    except Exception as e:
        # Handle other potential errors
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
        response.raise_for_status()  # Raises HTTPError for 4XX/5XX status
        response = response.json()  # convert to dict from string
        response = Box(response)  # convert to 'dot-dict' style which is the openai python client default
        return response
    except requests.exceptions.HTTPError as http_err:
        # Handle HTTP errors (e.g., response 4XX, 5XX)
        raise http_err
    except requests.exceptions.RequestException as req_err:
        # Handle other requests-related errors (e.g., connection error)
        raise req_err
    except Exception as e:
        # Handle other potential errors
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
                    printd(f"Got a rate limit error ('{http_err}') on LLM backend request, waiting {int(delay)}s then retrying...")
                    time.sleep(delay)
                else:
                    # For other HTTP errors, re-raise the exception
                    raise

            # Raise exceptions for any errors not specified
            except Exception as e:
                raise e

    return wrapper


# TODO: delete/ignore --legacy
@retry_with_exponential_backoff
def completions_with_backoff(**kwargs):
    # Local model
    if HOST_TYPE is not None:
        return get_chat_completion(**kwargs)

    # OpenAI / Azure model
    else:
        if using_azure():
            azure_openai_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
            if azure_openai_deployment is not None:
                kwargs["deployment_id"] = azure_openai_deployment
            else:
                kwargs["engine"] = MODEL_TO_AZURE_ENGINE[kwargs["model"]]
                kwargs.pop("model")
        if "context_window" in kwargs:
            kwargs.pop("context_window")

        api_url = "https://api.openai.com/v1"
        api_key = os.get_env("OPENAI_API_KEY")
        if api_key is None:
            raise Exception("OPENAI_API_KEY is not defined - please set it")
        return openai_chat_completions_request(api_url, api_key, data=kwargs)


@retry_with_exponential_backoff
def chat_completion_with_backoff(
    agent_config,
    messages,
    functions,
    function_call,
):
    from memgpt.utils import printd
    from memgpt.config import MemGPTConfig

    config = MemGPTConfig.load()  # load credentials (currently not stored in agent config)

    printd(f"Using model {agent_config.model_endpoint_type}, endpoint: {agent_config.model_endpoint}")
    if agent_config.model_endpoint_type == "openai":
        # openai
        return openai_chat_completions_request(
            url=agent_config.model_endpoint,  # https://api.openai.com/v1 -> https://api.openai.com/v1/chat/completions
            api_key=config.openai_key,  # 'sk....'
            data=dict(
                model=agent_config.model,
                messages=messages,
                functions=functions,
                function_call=function_call,
                user=config.anon_clientid,
            ),
        )
    elif agent_config.model_endpoint_type == "azure":
        # azure
        azure_deployment = config.azure_deployment if config.azure_deployment is not None else MODEL_TO_AZURE_ENGINE[agent_config.model]
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
            model=agent_config.model,
            messages=messages,
            functions=functions,
            function_call=function_call,
            context_window=agent_config.context_window,
            endpoint=agent_config.model_endpoint,
            endpoint_type=agent_config.model_endpoint_type,
            wrapper=agent_config.model_wrapper,
            user=config.anon_clientid,
        )


# TODO: deprecate
@retry_with_exponential_backoff
def create_embedding_with_backoff(**kwargs):
    if using_azure():
        azure_openai_deployment = os.getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT")
        if azure_openai_deployment is not None:
            kwargs["deployment_id"] = azure_openai_deployment
        else:
            kwargs["engine"] = kwargs["model"]
            kwargs.pop("model")

        api_key = os.get_env("AZURE_OPENAI_KEY")
        if api_key is None:
            raise Exception("AZURE_OPENAI_API_KEY is not defined - please set it")
        # TODO check
        # api_version???
        # resource_name???
        # "engine" instead of "model"???
        return azure_openai_embeddings_request(
            resource_name=None, deployment_id=azure_openai_deployment, api_version=None, api_key=api_key, data=kwargs
        )

    else:
        # return openai.Embedding.create(**kwargs)
        api_url = "https://api.openai.com/v1"
        api_key = os.get_env("OPENAI_API_KEY")
        if api_key is None:
            raise Exception("OPENAI_API_KEY is not defined - please set it")
        return openai_embeddings_request(url=api_url, api_key=api_key, data=kwargs)


def get_embedding_with_backoff(text, model="text-embedding-ada-002"):
    text = text.replace("\n", " ")
    response = create_embedding_with_backoff(input=[text], model=model)
    embedding = response["data"][0]["embedding"]
    return embedding


MODEL_TO_AZURE_ENGINE = {
    "gpt-4-1106-preview": "gpt-4-1106-preview",  # TODO check
    "gpt-4": "gpt-4",
    "gpt-4-32k": "gpt-4-32k",
    "gpt-3.5": "gpt-35-turbo",  # diff
    "gpt-3.5-turbo": "gpt-35-turbo",  # diff
    "gpt-3.5-turbo-16k": "gpt-35-turbo-16k",  # diff
}


def get_set_azure_env_vars():
    azure_env_variables = [
        ("AZURE_OPENAI_KEY", os.getenv("AZURE_OPENAI_KEY")),
        ("AZURE_OPENAI_ENDPOINT", os.getenv("AZURE_OPENAI_ENDPOINT")),
        ("AZURE_OPENAI_VERSION", os.getenv("AZURE_OPENAI_VERSION")),
        ("AZURE_OPENAI_DEPLOYMENT", os.getenv("AZURE_OPENAI_DEPLOYMENT")),
        (
            "AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT",
            os.getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT"),
        ),
    ]
    return [x for x in azure_env_variables if x[1] is not None]


def using_azure():
    return len(get_set_azure_env_vars()) > 0


def configure_azure_support():
    azure_openai_key = os.getenv("AZURE_OPENAI_KEY")
    azure_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    azure_openai_version = os.getenv("AZURE_OPENAI_VERSION")
    if None in [
        azure_openai_key,
        azure_openai_endpoint,
        azure_openai_version,
    ]:
        print(f"Error: missing Azure OpenAI environment variables. Please see README section on Azure.")
        return


def check_azure_embeddings():
    azure_openai_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
    azure_openai_embedding_deployment = os.getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT")
    if azure_openai_deployment is not None and azure_openai_embedding_deployment is None:
        raise ValueError(
            f"Error: It looks like you are using Azure deployment ids and computing embeddings, make sure you are setting one for embeddings as well. Please see README section on Azure"
        )
