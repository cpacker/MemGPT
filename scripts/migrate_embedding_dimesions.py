from memgpt.config import MemGPTConfig
from sqlalchemy.engine import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy import text


def update_pgvector_dim(uri: str, table_name: str) -> None:
    """
    connects to a postgres instance and reduces the dimensions of the embedding column to 1536
    """
    engine = create_engine(uri)
    session_maker = sessionmaker(bind=engine)
    with session_maker() as session:
        session.execute(text(f"ALTER TABLE {table_name} ADD COLUMN embedding_new vector(1536);"))
        session.execute(text(f"UPDATE {table_name} SET embedding_new = CAST((CAST(embedding AS real[]))[1:1536] AS vector);"))
        session.execute(text(f"ALTER TABLE {table_name} DROP COLUMN embedding;"))
        session.execute(text(f"ALTER TABLE {table_name} rename embedding_new TO embedding;"))
        session.commit()


if __name__ == "__main__":

    memgpt_config: MemGPTConfig = MemGPTConfig.load()

    if memgpt_config.archival_storage_type == "postgres":
        archival_table_name = "memgpt_archival_memory_agent"
        archival_uri = memgpt_config.archival_storage_uri
        print(f'Updating {archival_table_name} - setting dimensions for column "embedding" to 1536')
        update_pgvector_dim(uri=archival_uri, table_name=archival_table_name)

    if memgpt_config.recall_storage_type == "postgres":
        recall_table_name = "memgpt_recall_memory_agent"
        recall_uri = memgpt_config.recall_storage_uri
        print(f'Updating {recall_table_name} - setting dimensions for column "embedding" to 1536')
        update_pgvector_dim(uri=recall_uri, table_name=recall_table_name)

        passages_table_name = "memgpt_passages"
        print(f'Updating {passages_table_name} - setting dimensions for column "embedding" to 1536')
        update_pgvector_dim(uri=recall_uri, table_name=passages_table_name)
