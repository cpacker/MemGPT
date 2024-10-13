import os

from letta.providers import (
    AnthropicProvider,
    GoogleAIProvider,
    OllamaProvider,
    OpenAIProvider,
)
from letta.settings import model_settings


def test_openai():
    api_key = os.getenv("OPENAI_API_KEY")
    assert api_key is not None
    provider = OpenAIProvider(api_key=api_key, base_url=model_settings.openai_api_base)
    models = provider.list_llm_models()
    print(models)


def test_anthropic():
    api_key = os.getenv("ANTHROPIC_API_KEY")
    assert api_key is not None
    provider = AnthropicProvider(api_key=api_key)
    models = provider.list_llm_models()
    print(models)


# def test_groq():
#    provider = GroqProvider(api_key=os.getenv("GROQ_API_KEY"))
#    models = provider.list_llm_models()
#    print(models)
#
#


# TODO: Add this test
# https://linear.app/letta/issue/LET-159/add-tests-for-azure-openai-in-test-providerspy-and-test-endpointspy
def test_azure():
    pass


def test_ollama():
    base_url = os.getenv("OLLAMA_BASE_URL")
    assert base_url is not None
    provider = OllamaProvider(base_url=base_url, default_prompt_formatter=model_settings.default_prompt_formatter, api_key=None)
    models = provider.list_llm_models()
    print(models)

    embedding_models = provider.list_embedding_models()
    print(embedding_models)


def test_googleai():
    api_key = os.getenv("GEMINI_API_KEY")
    assert api_key is not None
    provider = GoogleAIProvider(api_key=api_key)
    models = provider.list_llm_models()
    print(models)

    provider.list_embedding_models()


# def test_vllm():
#    provider = VLLMProvider(base_url=os.getenv("VLLM_API_BASE"))
#    models = provider.list_llm_models()
#    print(models)
#
#    provider.list_embedding_models()
