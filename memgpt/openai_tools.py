import random
import os
import time

import time
from typing import Callable, TypeVar

from memgpt.local_llm.chat_completion_proxy import get_chat_completion

HOST = os.getenv("OPENAI_API_BASE")
HOST_TYPE = os.getenv("BACKEND_TYPE")  # default None == ChatCompletion
R = TypeVar("R")

import openai

if HOST is not None:
    openai.api_base = HOST


def retry_with_exponential_backoff(
    func,
    initial_delay: float = 1,
    exponential_base: float = 2,
    jitter: bool = True,
    max_retries: int = 20,
    errors: tuple = (openai.error.RateLimitError,),
):
    """Retry a function with exponential backoff."""

    def wrapper(*args, **kwargs):
        # Initialize variables
        num_retries = 0
        delay = initial_delay

        # Loop until a successful response or max_retries is hit or an exception is raised
        while True:
            try:
                return func(*args, **kwargs)

            # Retry on specified errors
            except errors as e:
                # Increment retries
                num_retries += 1

                # Check if max retries has been reached
                if num_retries > max_retries:
                    raise Exception(f"Maximum number of retries ({max_retries}) exceeded.")

                # Increment the delay
                delay *= exponential_base * (1 + jitter * random.random())

                # Sleep for the delay
                time.sleep(delay)

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
        return openai.ChatCompletion.create(**kwargs)


@retry_with_exponential_backoff
def chat_completion_with_backoff(agent_config, **kwargs):
    from memgpt.utils import printd
    from memgpt.config import AgentConfig, MemGPTConfig

    # both "model" and "messages" are required for base OpenAI calls
    # also required for local LLM Ollama, but not others
    if "model" not in kwargs:
        kwargs["model"] = agent_config.model

    printd(f"Using model {agent_config.model_endpoint_type}, endpoint: {agent_config.model_endpoint}")
    if agent_config.model_endpoint_type == "openai":
        # openai
        openai.api_base = agent_config.model_endpoint
        return openai.ChatCompletion.create(**kwargs)
    elif agent_config.model_endpoint_type == "azure":
        # configure openai
        config = MemGPTConfig.load()  # load credentials (currently not stored in agent config)
        openai.api_type = "azure"
        openai.api_key = config.azure_key
        openai.api_base = config.azure_endpoint
        openai.api_version = config.azure_version
        if config.azure_deployment is not None:
            kwargs["deployment_id"] = config.azure_deployment
        else:
            kwargs["engine"] = MODEL_TO_AZURE_ENGINE[config.model]
            del kwargs["model"]
        return openai.ChatCompletion.create(**kwargs)
    else:  # local model
        kwargs["context_window"] = agent_config.context_window  # specify for open LLMs
        kwargs["endpoint"] = agent_config.model_endpoint  # specify for open LLMs
        kwargs["endpoint_type"] = agent_config.model_endpoint_type  # specify for open LLMs
        kwargs["wrapper"] = agent_config.model_wrapper  # specify for open LLMs
        return get_chat_completion(**kwargs)


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
    return openai.Embedding.create(**kwargs)


def get_embedding_with_backoff(text, model="text-embedding-ada-002"):
    text = text.replace("\n", " ")
    response = create_embedding_with_backoff(input=[text], model=model)
    embedding = response["data"][0]["embedding"]
    return embedding


MODEL_TO_AZURE_ENGINE = {
    "gpt-4": "gpt-4",
    "gpt-4-32k": "gpt-4-32k",
    "gpt-3.5": "gpt-35-turbo",
    "gpt-3.5-turbo": "gpt-35-turbo",
    "gpt-3.5-turbo-16k": "gpt-35-turbo-16k",
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

    openai.api_type = "azure"
    openai.api_key = azure_openai_key
    openai.api_base = azure_openai_endpoint
    openai.api_version = azure_openai_version
    # deployment gets passed into chatcompletion


def check_azure_embeddings():
    azure_openai_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
    azure_openai_embedding_deployment = os.getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT")
    if azure_openai_deployment is not None and azure_openai_embedding_deployment is None:
        raise ValueError(
            f"Error: It looks like you are using Azure deployment ids and computing embeddings, make sure you are setting one for embeddings as well. Please see README section on Azure"
        )
