import os
import requests
import base64

from tests.helpers.endpoints_helper import check_vision_input

llm_config_dir = "tests/configs/llm_model_configs"

IMAGE_URL_DOG = "https://upload.wikimedia.org/wikipedia/commons/7/76/Whippet_2018_6.jpg"

def test_vision_openai():
    filename = os.path.join(llm_config_dir, "openai-gpt-4o.json")
    response = check_vision_input(filename,
                                  image_url=IMAGE_URL_DOG,
                                  keyword="dog")
    print(f"Got successful response from client: \n\n{response}")

def test_vision_openai_base64():
    image_get_response = requests.get(IMAGE_URL_DOG)
    image_get_response.raise_for_status()
    base64_encoded_image = base64.b64encode(image_get_response.content).decode('utf-8')

    filename = os.path.join(llm_config_dir, "openai-gpt-4o.json")
    response = check_vision_input(filename,
                                  image_url=f"data:image/jpeg;base64,{base64_encoded_image}",
                                  keyword="dog")
    print(f"Got successful response from client: \n\n{response}")