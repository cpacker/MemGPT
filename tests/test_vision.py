import os
import requests
import base64

from tests.helpers.endpoints_helper import check_vision_input

llm_config_dir = "tests/configs/llm_model_configs"

IMAGE_URL_DOG = "https://images.unsplash.com/photo-1625992095709-b74d6be8b422?q=80&w=2532&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D"
IMAGE_URL_CAT = "https://images.unsplash.com/photo-1607281671197-399dd9d01af5?q=80&w=2468&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D"

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

def test_vision_openai_multiple_images():
    filename = os.path.join(llm_config_dir, "openai-gpt-4o.json")
    response = check_vision_input(filename,
                                  image_url=[IMAGE_URL_DOG, IMAGE_URL_CAT],
                                  keyword="dog")
    print(f"Got successful response from client: \n\n{response}")