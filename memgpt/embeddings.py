import typer
from llama_index.embeddings import OpenAIEmbedding


def embedding_model():
    """Return LlamaIndex embedding model to use for embeddings"""

    from memgpt.config import MemGPTConfig

    # load config
    config = MemGPTConfig.load()

    # TODO: use embedding_endpoint in the future
    endpoint = config.embedding_model
    if endpoint == "openai":
        model = OpenAIEmbedding(api_base="https://api.openai.com/v1", api_key=config.openai_key)
        return model
    elif endpoint == "azure":
        return OpenAIEmbedding(
            model="text-embedding-ada-002",
            deployment_name=config.azure_embedding_deployment,
            api_key=config.azure_key,
            api_base=config.azure_endpoint,
            api_type="azure",
            api_version=config.azure_version,
        )
    elif endpoint == "local":
        # default to hugging face model
        from llama_index.embeddings import HuggingFaceEmbedding

        model = "BAAI/bge-small-en-v1.5"
        typer.secho(
            f"Warning: defaulting to HuggingFace embedding model {model} since model endpoint is not OpenAI or Azure.",
            fg=typer.colors.YELLOW,
        )
        typer.secho(f"Warning: ensure torch and transformers are installed")
        # return f"local:{model}"

        # loads BAAI/bge-small-en-v1.5
        return HuggingFaceEmbedding(model_name=model)
    else:
        # use env variable OPENAI_API_BASE
        model = OpenAIEmbedding()
        return model
