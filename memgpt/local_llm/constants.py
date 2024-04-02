# import memgpt.local_llm.llm_chat_completion_wrappers.airoboros as airoboros
from memgpt.local_llm.llm_chat_completion_wrappers.chatml import ChatMLInnerMonologueWrapper, ChatMLOuterInnerMonologueWrapper

DEFAULT_ENDPOINTS = {
    # Local
    "koboldcpp": "http://localhost:5001",
    "llamacpp": "http://localhost:8080",
    "lmstudio": "http://localhost:1234",
    "lmstudio-legacy": "http://localhost:1234",
    "ollama": "http://localhost:11434",
    "webui-legacy": "http://localhost:5000",
    "webui": "http://localhost:5000",
    "vllm": "http://localhost:8000",
    # APIs
    "openai": "https://api.openai.com",
    "groq": "https://api.groq.com/openai",
}

DEFAULT_OLLAMA_MODEL = "dolphin2.2-mistral:7b-q6_K"

# DEFAULT_WRAPPER = airoboros.Airoboros21InnerMonologueWrapper
# DEFAULT_WRAPPER_NAME = "airoboros-l2-70b-2.1"

DEFAULT_WRAPPER = ChatMLInnerMonologueWrapper
DEFAULT_WRAPPER_NAME = "chatml"
