import typer
from llama_index.embeddings import OpenAIEmbedding


def embedding_model():
    """Return LlamaIndex embedding model to use for embeddings"""

    from memgpt.config import MemGPTConfig

    # load config
    config = MemGPTConfig.load()

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


class Index:
    def __init__(self, name: str, save_directory: Optional[str] = None):

        config = MemGPTConfig.load()
        self.save_directory = save_directory

        # setup storage
        self.storage_type = config.archival_storage_type
        print("VECTORDB CONFIG", self.save_directory, self.storage_type)
        if config.archival_storage_type == "local":
            self.storage_context = StorageContext.from_defaults(persist_dir=self.save_directory)
        else:
            if config.archival_storage_type == "postgres":
                from llama_index.vector_stores import PGVectorStore

                self.vector_store = PGVectorStore.from_params(
                    database=self.db_name,
                    host=url.host,
                    password=url.password,
                    port=url.port,
                    user=url.username,
                    table_name=name,  # table_name = data source name
                    embed_dim=MemGPTConfig.load().embedding_dim,  # openai embedding dimension
                )
            elif config.archival_storage_type == "chroma":
                from llama_index.vector_stores import ChromaVectorStore
                import chromadb

                print("use chroma")
                # chroma_client = chromadb.EphemeralClient()
                chroma_client = chromadb.PersistentClient(path="/Users/sarahwooders/repos/MemGPT/chromadb")
                chroma_collection = chroma_client.get_or_create_collection(name)
                self.vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
            else:
                raise ValueError(f"Unknown archival storage type {config.archival_storage_type}")
            self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)

        # setup embedding model
        self.embed_model = embedding_model(config)

        # setup service context
        self.service_context = ServiceContext.from_defaults(llm=None, embed_model=self.embed_model, chunk_size=config.embedding_chunk_size)

    def load_documents(self, documents):
        self.index = VectorStoreIndex.from_documents(
            documents, storage_context=self.storage_context, service_context=self.service_context, show_progress=True
        )
        if self.storage_type == "local":
            # save to disk if local
            self.index.storage_context.persist(persist_dir=self.directory)  # TODO:

    def load_index(self, index_dir: str):
        storage_context = StorageContext.from_defaults(persist_dir=index_dir)
        self.index = load_index_from_storage(storage_context)

        # persist

    def get_index(self):
        if self.index:
            # index already loaded
            return self.index

        if self.storage_type == "local":
            self.index = load_index_from_storage(self.storage_context)
        else:
            self.index = VectorStoreIndex.from_vector_store(vector_store=self.vector_store)

        return self.index
