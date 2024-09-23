from letta.cli.cli import QuickstartChoice, quickstart
from letta.config import LettaConfig


def test_quickstart():

    # openai
    quickstart(QuickstartChoice.openai, debug=True, terminal=False)
    config = LettaConfig.load()
    assert config.default_llm_config.model_endpoint_type == "openai"
    assert config.default_embedding_config.embedding_endpoint_type == "openai"

    # letta
    quickstart(QuickstartChoice.letta_hosted, debug=True, terminal=False)
    config = LettaConfig.load()
    assert config.default_llm_config.model_endpoint == "https://inference.letta.ai"
    assert config.default_embedding_config.embedding_endpoint == "https://embeddings.letta.ai"
