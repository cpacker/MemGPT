import json
import os
import uuid
import requests
import base64
from typing import Optional

from letta import create_client
from letta.agent import Agent
from letta.config import LettaConfig
from letta.llm_api.llm_api_tools import create
from letta.schemas.embedding_config import EmbeddingConfig
from letta.schemas.llm_config import LLMConfig
from letta.schemas.message import Message, MultimodalMessage, ContentPart
# defaults (letta hosted)
embedding_config_path = "configs/embedding_model_configs/letta-hosted.json"
llm_config_path = "configs/llm_model_configs/letta-hosted.json"

# directories
embedding_config_dir = "configs/embedding_model_configs"
llm_config_dir = "configs/llm_model_configs"

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')
    
def b64_image_to_url(b64_image: str) -> str:
    return f"data:image/jpeg;base64,{b64_image}"

def convert_message_to_image_message(message: Message, image_base64: Optional[str] = None) -> MultimodalMessage:
    """
    Convert a Message object to an MultimodalMessage object.

    Args:
        message (Message): The original Message object to convert.
        image_base64 (Optional[str]): The base64-encoded image data, if available.

    Returns:
        MultimodalMessage: A new MultimodalMessage object with the data from the original Message.
    """
    # Create the content list for MultimodalMessage
    content = []
    
    # Add text content if available
    if message.text:
        content.append({
            "type": "text", 
            "text": message.text
        })
    
    # Add image content if available
    if image_base64 and message.role == "user":
        content.append({
            "type": "image_url",
            "image_url": {
                "url": b64_image_to_url(image_base64)
            }
        })
    
    # Create and return the MultimodalMessage
    return MultimodalMessage(
        id=message.id,
        role=message.role,
        content=content,
        text=message.text,
        user_id=message.user_id,
        agent_id=message.agent_id,
        model=message.model,
        name=message.name,
        created_at=message.created_at,
        tool_calls=message.tool_calls,
        tool_call_id=message.tool_call_id
    )

def run_llm_endpoint(filename):
    config_data = json.load(open(filename, "r"))
    print(config_data)
    llm_config = LLMConfig(**config_data)
    embedding_config = EmbeddingConfig(**json.load(open(embedding_config_path)))

    # setup config
    config = LettaConfig()
    config.default_llm_config = llm_config
    config.default_embedding_config = embedding_config
    config.save()

    client = create_client()
    agent_state = client.create_agent(name="test_agent", llm_config=llm_config, embedding_config=embedding_config)
    tools = [client.get_tool(client.get_tool_id(name=name)) for name in agent_state.tools]
    agent = Agent(
        interface=None,
        tools=tools,
        agent_state=agent_state,
        # gpt-3.5-turbo tends to omit inner monologue, relax this requirement for now
        first_message_verify_mono=True,
    )

    response = create(
        llm_config=llm_config,
        user_id=uuid.UUID(int=1),  # dummy user_id
        # messages=agent_state.messages,
        messages=agent._messages,
        functions=agent.functions,
        functions_python=agent.functions_python,
    )
    client.delete_agent(agent_state.id)
    assert response is not None

def test_llm_endpoint_openai():
    filename = os.path.join(llm_config_dir, "gpt-4.json")
    run_llm_endpoint(filename)
    print("test_llm_endpoint_openai passed")


def run_vision_endpoint_openai(filename, vision_model=False):
    config_data = json.load(open(filename, "r"))
    print(config_data)
    llm_config = LLMConfig(**config_data)
    embedding_config = EmbeddingConfig(**json.load(open(embedding_config_path)))

    # setup config
    config = LettaConfig()
    config.default_llm_config = llm_config
    config.default_embedding_config = embedding_config
    config.save()

    client = create_client()
    agent_state = client.create_agent(name="test_agent", llm_config=llm_config, embedding_config=embedding_config)
    tools = [client.get_tool(client.get_tool_id(name=name)) for name in agent_state.tools]
    agent = Agent(
        interface=None,
        tools=tools,
        agent_state=agent_state,
        # gpt-3.5-turbo tends to omit inner monologue, relax this requirement for now
        first_message_verify_mono=True,
    )

    # Getting the base64 string
    if vision_model:
        image_path = "/Users/themindinator/interview/memgpt/MemGPT/panda.jpg"
        base64_image = encode_image(image_path)
    else:
        base64_image = None

    print(f"MESSAGES {len(agent._messages)}", [m.role for m in agent._messages])
    img_messages = [convert_message_to_image_message(message, base64_image) for message in agent._messages]

    try:
        response = create(
            llm_config=llm_config,
            user_id=uuid.UUID(int=1),  # dummy user_id
            messages=img_messages,
            functions=agent.functions,
            functions_python=agent.functions_python,
        )
        client.delete_agent(agent_state.id)
    except Exception as e:
        print(f"ERROR {e}")
        client.delete_agent(agent_state.id)
        raise e
    print(f"RESPONSE {response}")
    assert response is not None

def run_vision_client_text():
    client = create_client()
    agent_state = client.create_agent(name="test_agent")
    
    try:
        response = client.send_message(
            agent_id=agent_state.id,
            role="user",
            message=ContentPart(type="text", text="hello")
        )
        print("Usage", response.usage)
        print("Agent messages", response.messages)
    except Exception as e:
        print(f"ERROR {e}")
        client.delete_agent(agent_state.id)
        raise e
    
    client.delete_agent(agent_state.id)
    assert response is not None

def run_vision_client_image():
    client = create_client()
    agent_state = client.create_agent(name="test_agent")
    image_path = "/Users/themindinator/interview/memgpt/MemGPT/panda.jpg"
    base64_image = encode_image(image_path)
    try:
        response = client.send_message(
            agent_id=agent_state.id,
            role="user",
            message=ContentPart(type="image_url", image_url={"url": b64_image_to_url(base64_image)}))
    except Exception as e:
        print(f"ERROR {e}")
        client.delete_agent(agent_state.id)
        raise e
    client.delete_agent(agent_state.id)
    assert response is not None

def test_vision_endpoint_openai_gpt4():
    filename = os.path.join(llm_config_dir, "gpt-4.json")
    run_vision_endpoint_openai(filename)
    print("test_vision_endpoint_openai gpt4 passed")

def test_vision_endpoint_openai_gpt4o():
    filename = os.path.join(llm_config_dir, "gpt-4o.json")
    run_vision_endpoint_openai(filename, vision_model=True)
    print("test_vision_endpoint_openai gpt4o passed")

def test_vision_client_text():
    run_vision_client_text()
    print("test_vision_client_text passed")

def test_vision_client_image():
    run_vision_client_image()
    print("test_vision_client_image passed")


def test_openai_with_images():
    # The API endpoint
    url = "https://api.openai.com/v1/chat/completions"
    image_path = "/Users/themindinator/interview/memgpt/MemGPT/panda.jpg"
    
    # Getting the base64 string
    base64_image = encode_image(image_path)

    # Forming the payload
    payload = {
        "model": "gpt-4o",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "What's in this image?"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 300
    }

    # Setting up the headers
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    # Making the request
    response = requests.post(url, headers=headers, json=payload)

    # Checking the response
    if response.status_code == 200:
        print(json.dumps(response.json(), indent=4))
    else:
        print(f"Request failed with status code: {response.status_code}")
        print(response.text)
    
    assert response is not None
    print("test_openai_with_images passed")

if __name__ == "__main__":
    # test_llm_endpoint_openai()
    # test_openai_with_images()
    # test_vision_endpoint_openai_gpt4()
    # test_vision_endpoint_openai_gpt4o()
    # print('-'*100)
    # test_vision_client_text()
    # print('-'*100)
    test_vision_client_image()