import requests

from letta.utils import printd, smart_urljoin

# https://docs.anthropic.com/claude/docs/models-overview
# Sadly hardcoded
MODEL_LIST = [
    {
        "name": "claude-3-opus-20240229",
        "context_window": 200000,
    },
    {
        "name": "claude-3-sonnet-20240229",
        "context_window": 200000,
    },
    {
        "name": "claude-3-haiku-20240307",
        "context_window": 200000,
    },
]

DUMMY_FIRST_USER_MESSAGE = "User initializing bootup sequence."


# def antropic_get_model_context_window(url: str, api_key: Union[str, None], model: str, ) -> int:
#     for model_dict in anthropic_get_model_list(url=url, api_key=api_key):
#         if model_dict["name"] == model:
#             return model_dict["context_window"]
#     raise ValueError(f"Can't find model '{model}' in Anthropic model list")


def mistral_get_model_list(url: str, api_key: str) -> dict:
    url = smart_urljoin(url, "models")

    headers = {"Content-Type": "application/json"}
    if api_key is not None:
        headers["Authorization"] = f"Bearer {api_key}"

    printd(f"Sending request to {url}")
    response = None
    try:
        # TODO add query param "tool" to be true
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Raises HTTPError for 4XX/5XX status
        response_json = response.json()  # convert to dict from string
        return response_json
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
