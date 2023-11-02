import typer
from llama_index.embeddings import OpenAIEmbedding
from llama_index.schema import BaseComponent, TextNode, Document


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
        print("storage type", self.storage_type)
        if config.archival_storage_type == "local":
            self.storage_context = StorageContext.from_defaults(persist_dir=self.save_directory)
        else:
            if config.archival_storage_type == "postgres":
                from llama_index.vector_stores import PGVectorStore
                from sqlalchemy import make_url

                connection_string = config.archival_storage_uri
                url = make_url(connection_string)

                self.vector_store = PGVectorStore.from_params(
                    database=url.database,
                    host=url.host,
                    password=url.password,
                    port=url.port,
                    user=url.username,
                    table_name=name,  # table_name = data source name
                    embed_dim=config.embedding_dim,  # openai embedding dimension
                )
                self.uri = config.archival_storage_uri
                self.table_name = "data_%s" % name.lower()  # TODO: figure out exactly what this is
                print("TABLE NAME", self.table_name)
            elif config.archival_storage_type == "chroma":
                from llama_index.vector_stores import ChromaVectorStore
                import chromadb

                # chroma_client = chromadb.EphemeralClient()
                # TODO: connect to storage URI if provided
                chroma_client = chromadb.PersistentClient(path="/Users/sarahwooders/repos/MemGPT/chromadb")
                chroma_collection = chroma_client.get_or_create_collection(name)
                self.vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
            else:
                raise ValueError(f"Unknown archival storage type {config.archival_storage_type}")
            self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
            print("storage context", self.storage_context)

        # setup embedding model
        self.embed_model = embedding_model(config)

        # setup service context
        self.service_context = ServiceContext.from_defaults(llm=None, embed_model=self.embed_model, chunk_size=config.embedding_chunk_size)

        # load index (if exists)
        # TODO: make sure this doesn't cause an error if the index doesn't exist yet
        if self.storage_type == "local":
            # load from disk if local
            self.index = load_index_from_storage(self.storage_context)
        else:
            # load from vector store
            self.index = VectorStoreIndex.from_vector_store(vector_store=self.vector_store)
            # print("BASIC QUERY", self.index.as_query_engine().query("cinderella crying"))

    def load_nodes(self, nodes):
        """Loads a list of LlamaIndex nodes into index

        :param nodes: List of nodes to create an index with
        :type nodes: List[TextNode]
        """
        # for node in nodes:
        #    node.text = node.text.replace("\x00", "\uFFFD")  # hacky fix for error on null characters
        self.index.build_index_from_nodes(nodes=nodes)
        print(f"Added {len(nodes)} nodes")
        self.persist()

    def load_documents(self, documents):
        """Load a list of documents into an index

        :param documents: List of documents to create an index with
        :type documents: List[Document]
        """
        # need to remove problematic characters to avoid errors
        for doc in documents:
            doc.text = doc.text.replace("\x00", "\uFFFD")  # hacky fix for error on null characters

        # create index
        self.index = VectorStoreIndex.from_documents(
            documents, storage_context=self.storage_context, service_context=self.service_context, show_progress=True
        )
        self.persist()

    def persist(self):

        # persist state
        if self.storage_type == "local":
            # save to disk if local
            self.index.storage_context.persist(persist_dir=self.directory)  # TODO:
        else:
            self.index.storage_context.persist()

    def insert(self, text: str, embedding: Optional[List[float]] = None):
        """Insert new string into index

        :param text: String to insert into index
        :type text: str
        """
        if embedding is None:
            self.index.insert(text)
        else:
            self.index.insert(Document(text=text, embedding=embedding))

    def update(self, documents, embeddings=[]):
        """Update an index with new documents

        :param documents: List of documents to update an index with
        :type documents: List[Document]
        """
        # need to remove problematic characters to avoid errors
        for doc in documents:
            doc.text = doc.text.replace("\x00", "\uFFFD")

        # TODO: make sure document is persisted in the remote DB

        # TODO: allow for existing embeddings

    def get_nodes(self):
        """Get the list of nodes from an index (useful for moving data from one index to another)

        :return: Nodes contained in index
        :rtype: List[TextNode]
        """

        if self.storage_type == "local":
            embed_dict = self.index._vector_store._data.embedding_dict
            node_dict = self.index._docstore.docs

            nodes = []
            for node_id, node in node_dict.items():
                vector = embed_dict[node_id]
                node.embedding = vector
                nodes.append(node)
            return nodes
        elif self.storage_type == "postgres":
            from sqlalchemy import create_engine, MetaData, Table, select

            engine = create_engine(self.uri)
            metadata = MetaData()
            # data_table = Table(self.table_name, metadata, autoload_with=engine, schema='public')k
            print(self.vector_store._table_class)

            # Initialize a list to store the Node objects
            nodes = []

            # Start a connection to the database
            with engine.connect() as conn:
                # Select all data from the table
                select_stmt = select(self.vector_store._table_class)
                results = conn.execute(select_stmt).all()

                print(results[0])
                print("DATA", results[1].embedding, results[1].text)

                # Iterate over the rows to create Node objects
                for row in results:
                    # Assuming that 'text' is the document and 'embedding' is the binary representation of the embedding
                    # If 'embedding' is stored in a different format, you might need to adjust the code to handle it correctly
                    # import json
                    # document = json.loads(row[1])
                    try:

                        node = TextNode(
                            id_=row.node_id,
                            text=row.text,
                            metadata=row.metadata,
                            embedding=row.embedding,
                        )
                        print(node)
                        # node = Document(document=row.text, embedding=list(row.embedding))
                    except Exception as e:
                        print(row)
                        raise e
                    nodes.append(node)
            print("nodes", len(nodes))
            return nodes

        elif self.storage_type == "chroma":
            raise NotImplementedError("TODO")
        else:
            raise ValueError(f"Unknown archival storage type {self.storage_type}")
