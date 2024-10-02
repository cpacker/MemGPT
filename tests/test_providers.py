import os

from letta.providers import AnthropicProvider, OpenAIProvider


def test_openai():

    provider = OpenAIProvider(api_key=os.getenv("OPENAI_API_KEY"))
    models = provider.list_llm_models()
    print(models)


def test_anthropic():
    provider = AnthropicProvider(api_key=os.getenv("ANTHROPIC_API_KEY"))
    models = provider.list_llm_models()
    print(models)


test_openai()
test_anthropic()
