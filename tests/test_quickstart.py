from memgpt.cli.cli import QuickstartChoice, quickstart
from memgpt.config import MemGPTConfig


def test_quickstart():

    # openai
    quickstart(QuickstartChoice.openai, debug=True, terminal=False)
    config = MemGPTConfig.load()
    assert config.default_llm_config.model_endpoint_type == "openai"
    assert config.default_embedding_config.embedding_endpoint_type == "openai"

    # memgpt
    quickstart(QuickstartChoice.memgpt_hosted, debug=True, terminal=False)
    config = MemGPTConfig.load()
    assert config.default_llm_config.model_endpoint == "https://inference.memgpt.ai"
    assert config.default_embedding_config.embedding_endpoint == "https://embeddings.memgpt.ai"
