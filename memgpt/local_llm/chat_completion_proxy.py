"""MemGPT sends a ChatCompletion request

Under the hood, we use the functions argument to turn
"""


"""Key idea: create drop-in replacement for agent's ChatCompletion call that runs on an OpenLLM backend"""

import os
import json
import requests

from .webui_settings import DETERMINISTIC, SIMPLE
from .llm_chat_completion_wrappers import airoboros

HOST = os.getenv('OPENAI_API_BASE')
HOST_TYPE = os.getenv('BACKEND_TYPE')  # default None == ChatCompletion


class DotDict(dict):
    """Allow dot access on properties similar to OpenAI response object"""

    def __getattr__(self, attr):
        return self.get(attr)

    def __setattr__(self, key, value):
        self[key] = value


async def get_chat_completion(
        model,  # no model, since the model is fixed to whatever you set in your own backend
        messages,
        functions,
        function_call="auto",
    ):
    if function_call != "auto":
        raise ValueError(f"function_call == {function_call} not supported (auto only)")

    if True or model == 'airoboros_v2.1':
        llm_wrapper = airoboros.Airoboros21Wrapper()

    # First step: turn the message sequence into a prompt that the model expects
    prompt = llm_wrapper.chat_completion_to_prompt(messages, functions)
    # print(prompt)

    if HOST_TYPE != 'webui':
        raise ValueError(HOST_TYPE)

    request = SIMPLE
    request['prompt'] = prompt

    try:

        URI = f'{HOST}/v1/generate'
        response = requests.post(URI, json=request)
        if response.status_code == 200:
            # result = response.json()['results'][0]['history']
            result = response.json()
            # print(f"raw API response: {result}")
            result = result['results'][0]['text']
            print(f"json API response.text: {result}")
        else:
            raise Exception(f"API call got non-200 response code")

        # cleaned_result, chatcompletion_result = parse_st_json_output(result)
        chat_completion_result = llm_wrapper.output_to_chat_completion_response(result)
        print(json.dumps(chat_completion_result, indent=2))
        # print(cleaned_result)

        # unpack with response.choices[0].message.content
        response = DotDict({
            'model': None,
            'choices': [DotDict({
                'message': DotDict(chat_completion_result),
                'finish_reason': 'stop',  # TODO vary based on webui response
            })],
            'usage': DotDict({
                # TODO fix
                'prompt_tokens': 0,
                'completion_tokens': 0,
                'total_tokens': 0,
            })
        })
        return response

    except Exception as e:
        # TODO
        raise e
