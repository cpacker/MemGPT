import os

from letta.providers import (
    AnthropicProvider,
    AzureProvider,
    GoogleAIProvider,
    GroqProvider,
    MistralProvider,
    OllamaProvider,
    OpenAIProvider,
    TogetherProvider,
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


def test_groq():
    provider = GroqProvider(api_key=os.getenv("GROQ_API_KEY"))
    models = provider.list_llm_models()
    print(models)


def test_azure():
    provider = AzureProvider(api_key=os.getenv("AZURE_API_KEY"), base_url=os.getenv("AZURE_BASE_URL"))
    models = provider.list_llm_models()
    print([m.model for m in models])

    embed_models = provider.list_embedding_models()
    print([m.embedding_model for m in embed_models])


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


def test_mistral():
    provider = MistralProvider(api_key=os.getenv("MISTRAL_API_KEY"))
    models = provider.list_llm_models()
    print([m.model for m in models])


def test_together():
    provider = TogetherProvider(api_key=os.getenv("TOGETHER_API_KEY"), default_prompt_formatter="chatml")
    models = provider.list_llm_models()
    print([m.model for m in models])

    embedding_models = provider.list_embedding_models()
    print([m.embedding_model for m in embedding_models])


# def test_vllm():
#    provider = VLLMProvider(base_url=os.getenv("VLLM_API_BASE"))
#    models = provider.list_llm_models()
#    print(models)
#
#    provider.list_embedding_models()
