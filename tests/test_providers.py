import os

from letta.providers import AnthropicProvider, GroqProvider, OpenAIProvider


def test_openai():

    provider = OpenAIProvider(api_key=os.getenv("OPENAI_API_KEY"))
    models = provider.list_llm_models()
    print(models)


def test_anthropic():
    provider = AnthropicProvider(api_key=os.getenv("ANTHROPIC_API_KEY"))
    models = provider.list_llm_models()
    print(models)


def test_groq():
    provider = GroqProvider(api_key=os.getenv("GROQ_API_KEY"))
    models = provider.list_llm_models()
    print(models)


# test_groq()
test_openai()
# test_anthropic()
