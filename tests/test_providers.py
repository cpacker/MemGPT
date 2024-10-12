import os

from letta.providers import (
    AnthropicProvider,
    GoogleAIProvider,
    OllamaProvider,
    OpenAIProvider,
)


def test_openai():

    provider = OpenAIProvider(api_key=os.getenv("OPENAI_API_KEY"))
    models = provider.list_llm_models()
    print(models)


def test_anthropic():
    if os.getenv("ANTHROPIC_API_KEY") is None:
        return
    provider = AnthropicProvider(api_key=os.getenv("ANTHROPIC_API_KEY"))
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
    provider = OllamaProvider(base_url=os.getenv("OLLAMA_BASE_URL"))
    models = provider.list_llm_models()
    print(models)

    embedding_models = provider.list_embedding_models()
    print(embedding_models)


def test_googleai():
    provider = GoogleAIProvider(api_key=os.getenv("GEMINI_API_KEY"))
    models = provider.list_llm_models()
    print(models)

    provider.list_embedding_models()


# def test_vllm():
#    provider = VLLMProvider(base_url=os.getenv("VLLM_API_BASE"))
#    models = provider.list_llm_models()
#    print(models)
#
#    provider.list_embedding_models()
