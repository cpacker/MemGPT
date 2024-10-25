import os
import secrets
import string
import time
from typing import Optional

import trustme
import uvicorn
from fastapi import FastAPI

from letta.schemas.embedding_config import EmbeddingConfig
from letta.schemas.llm_config import LLMConfig

DEFAULT_MOCK_LLM_API_HOST = "localhost"
DEFAULT_MOCK_LLM_API_PORT = 8000
DEFAULT_MOCK_LLM_SSL_CERT_PATH = "certs/ca_cert.pem"


app = FastAPI()


@app.post("/v1/chat/completions")
async def user_message():
    response = {
        "id": "chatcmpl-" + generate_mock_id(28),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": "memgpt-openai",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_" + generate_mock_id(24),
                            "type": "function",
                            "function": {
                                "name": "send_message",
                                "arguments": '{"message":"Hello! It\'s great to meet you! How are you doing today?","inner_thoughts":"User has greeted me. Time to establish a connection and gauge their mood."}',
                            },
                        }
                    ],
                    "refusal": None,
                },
                "logprobs": None,
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {
            "prompt_tokens": 2370,
            "completion_tokens": 48,
            "total_tokens": 2418,
            "prompt_tokens_details": {"cached_tokens": 0},
            "completion_tokens_details": {"reasoning_tokens": 0},
        },
        "system_fingerprint": "fp_" + generate_mock_id(10),
    }
    return response


@app.get("/v1/embeddings")
async def message():
    pass


@app.get("/configs")
def get_config():
    return {
        "llm": get_llm_config(),
        "embedding": get_embedding_config(),
    }


def get_llm_config():
    return LLMConfig(
        model="memgpt-openai",
        model_endpoint_type="openai",
        model_endpoint="https://localhost:8000/v1",
        model_wrapper=None,
        context_window=8192,
    )


def get_embedding_config():
    return EmbeddingConfig(
        embedding_model="text-embedding-ada-002",
        embedding_endpoint_type="openai",
        embedding_endpoint="https://localhost:8000/v1",
        embedding_dim=1536,
        embedding_chunk_size=300,
    )


def generate_mock_id(length: int):
    possible_characters = string.ascii_letters + string.digits
    return "".join(secrets.choice(possible_characters) for _ in range(length))


def start_mock_llm_server(
    port: Optional[int] = DEFAULT_MOCK_LLM_API_PORT,
    host: Optional[str] = DEFAULT_MOCK_LLM_API_HOST,
):
    ca = trustme.CA()
    os.makedirs(DEFAULT_MOCK_LLM_SSL_CERT_PATH.split("/")[0], exist_ok=True)
    ca.cert_pem.write_to_path(DEFAULT_MOCK_LLM_SSL_CERT_PATH)
    os.environ["REQUESTS_CA_BUNDLE"] = DEFAULT_MOCK_LLM_SSL_CERT_PATH

    cert = ca.issue_cert(host)
    with cert.cert_chain_pems[0].tempfile() as cert_path:
        with cert.private_key_pem.tempfile() as key_path:
            print(f"Running: uvicorn server:mock_llm_app --host {host} --port {port}")
            uvicorn.run(
                app,
                host=host,
                port=port,
                ssl_keyfile=key_path,
                ssl_certfile=cert_path,
            )
