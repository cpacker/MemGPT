import os
from urllib.parse import urljoin
import requests
import json

# from .settings import SIMPLE

HOST = os.getenv("OPENAI_API_BASE")
HOST_TYPE = os.getenv("BACKEND_TYPE")  # default None == ChatCompletion
LMSTUDIO_API_SUFFIX = "/v1/chat/completions"
DEBUG = False

from .settings import SIMPLE


def get_lmstudio_completion(prompt, settings=SIMPLE):
    """Based on the example for using LM Studio as a backend from https://github.com/lmstudio-ai/examples/tree/main/Hello%2C%20world%20-%20OpenAI%20python%20client"""

    # Settings for the generation, includes the prompt + stop tokens, max length, etc
    request = settings

    message_structure = [{"role": "user", "content": prompt}]
    prompt_string = json.dumps(message_structure, indent=2)
    request["messages"] = message_structure
    
    if not HOST.startswith(("http://", "https://")):
        raise ValueError(f"Provided OPENAI_API_BASE value ({HOST}) must begin with http:// or https://")

    try:
        URI = urljoin(HOST.strip("/") + "/", LMSTUDIO_API_SUFFIX.strip("/"))
        response = requests.post(URI, json=request)
        if response.status_code == 200:
            result = response.json()            
            result = result["choices"][0]["message"]["content"]
            if DEBUG:
                print(f"json API response.text: {result}")
        else:
            raise Exception(
                f"API call got non-200 response code for address: {URI}. Make sure that the LM Studio local inference server is running and reachable at {URI}."
            )
    except:
        # TODO handle gracefully
        raise

    return result
