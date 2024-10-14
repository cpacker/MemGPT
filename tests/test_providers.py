import os

from letta.providers import (
    AnthropicProvider,
    AzureProvider,
    GoogleAIProvider,
    MistralProvider,
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


def test_azure():
    provider = AzureProvider(api_key=os.getenv("AZURE_API_KEY"), base_url=os.getenv("AZURE_BASE_URL"))
    models = provider.list_llm_models()
    print([m.model for m in models])

    embed_models = provider.list_embedding_models()
    print([m.embedding_model for m in embed_models])


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


def test_mistral():
    provider = MistralProvider(api_key=os.getenv("MISTRAL_API_KEY"))
    models = provider.list_llm_models()
    print([m.model for m in models])


# def test_vllm():
#    provider = VLLMProvider(base_url=os.getenv("VLLM_API_BASE"))
#    models = provider.list_llm_models()
#    print(models)
#
#    provider.list_embedding_models()
