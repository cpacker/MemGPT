import os

from letta.providers import (
    AnthropicProvider,
    AzureProvider,
    GoogleAIProvider,
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
# def test_ollama():
#    provider = OllamaProvider()
#    models = provider.list_llm_models()
#    print(models)
#
#
def test_googleai():
    provider = GoogleAIProvider(api_key=os.getenv("GEMINI_API_KEY"))
    models = provider.list_llm_models()
    print(models)

    provider.list_embedding_models()


def test_azure_openai():

    provider = AzureProvider(api_key=os.getenv("AZURE_API_KEY"), base_url=os.getenv("AZURE_BASE_URL"))
    provider.list_llm_models()
    embeds = provider.list_embedding_models()
    print(embeds)


#
#
# test_googleai()
# test_ollama()
# test_groq()
# test_openai()
# test_anthropic()
