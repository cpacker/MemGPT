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


@retry_with_exponential_backoff
def create(
    agent_config,
    messages,
    functions,
    function_call,
):

    """Return response to chat completion with backoff"""
    from memgpt.utils import printd
    from memgpt.config import MemGPTConfig

    config = MemGPTConfig.load()  # load credentials (currently not stored in agent config)

    printd(f"Using model {agent_config.model_endpoint_type}, endpoint: {agent_config.model_endpoint}")
    if agent_config.model_endpoint_type == "openai":
        # openai
        openai.api_base = agent_config.model_endpoint
        return openai.ChatCompletion.create(
            model=agent_config.model, messages=messages, functions=functions, function_call=function_call, user=config.anon_clientid
        )
    elif agent_config.model_endpoint_type == "azure":
        # azure
        openai.api_type = "azure"
        openai.api_key = config.azure_key
        openai.api_base = config.azure_endpoint
        openai.api_version = config.azure_version
        deployment_id = config.azure_deployment
        engine = None
        model = config.model
        return openai.ChatCompletion.create(
            model=model,
            messages=messages,
            engine=engine,
            deployment_id=deployment_id,
            functions=functions,
            function_call=function_call,
            user=config.anon_clientid,
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
