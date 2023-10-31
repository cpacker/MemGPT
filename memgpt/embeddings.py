from memgpt.config import MemGPTConfig
import typer
from llama_index.embeddings import OpenAIEmbedding


def embedding_model(config: MemGPTConfig):
    # TODO: use embedding_endpoint in the future
    if config.model_endpoint == "openai":
        return OpenAIEmbedding()
    elif config.model_endpoint == "azure":
        return OpenAIEmbedding(
            model="text-embedding-ada-002",
            deployment_name=config.azure_embedding_deployment,
            api_key=config.azure_key,
            api_base=config.azure_endpoint,
            api_type="azure",
            api_version=config.azure_version,
        )
    else:
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
