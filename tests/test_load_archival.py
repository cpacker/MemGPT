# import tempfile
# import asyncio
# import os
# import asyncio
# from datasets import load_dataset

# import memgpt
# from memgpt.cli.cli_load import load_directory, load_database, load_webpage
# import memgpt.presets as presets
# import memgpt.personas.personas as personas
# import memgpt.humans.humans as humans
# from memgpt.persistence_manager import InMemoryStateManager, LocalStateManager

# # from memgpt.config import AgentConfig
# from memgpt.constants import MEMGPT_DIR, DEFAULT_MEMGPT_MODEL
# import memgpt.interface  # for printing to terminal


def test_load_directory():
    return
    # downloading hugging face dataset (if does not exist)
    dataset = load_dataset("MemGPT/example_short_stories")

    cache_dir = os.getenv("HF_DATASETS_CACHE")

    if cache_dir is None:
        # Construct the default path if the environment variable is not set.
        cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "datasets")

    # load directory
    print("Loading dataset into index...")
    print(cache_dir)
    load_directory(
        name="tmp_hf_dataset",
        input_dir=cache_dir,
        recursive=True,
    )

    # create agents with defaults
    agent_config = AgentConfig(
        persona=personas.DEFAULT,
        human=humans.DEFAULT,
        model=DEFAULT_MEMGPT_MODEL,
        data_source="tmp_hf_dataset",
    )

    # create state manager based off loaded data
    persistence_manager = LocalStateManager(agent_config=agent_config)

    # create agent
    memgpt_agent = presets.use_preset(
        presets.DEFAULT_PRESET,
        agent_config,
        DEFAULT_MEMGPT_MODEL,
        personas.get_persona_text(personas.DEFAULT),
        humans.get_human_text(humans.DEFAULT),
        memgpt.interface,
        persistence_manager,
    )

    def query(q):
        res = asyncio.run(memgpt_agent.archival_memory_search(q))
        return res

    results = query("cinderella be getting sick")
    assert "Cinderella" in results, f"Expected 'Cinderella' in results, but got {results}"


def test_load_webpage():
    pass


def test_load_database():
    return
    from sqlalchemy import create_engine, MetaData
    import pandas as pd

    db_path = "memgpt/personas/examples/sqldb/test.db"
    engine = create_engine(f"sqlite:///{db_path}")

    # Create a MetaData object and reflect the database to get table information.
    metadata = MetaData()
    metadata.reflect(bind=engine)

    # Get a list of table names from the reflected metadata.
    table_names = metadata.tables.keys()

    print(table_names)

    # Define a SQL query to retrieve data from a table (replace 'your_table_name' with your actual table name).
    query = f"SELECT * FROM {list(table_names)[0]}"

    # Use Pandas to read data from the database into a DataFrame.
    df = pd.read_sql_query(query, engine)
    print(df)

    load_database(
        name="tmp_db_dataset",
        # engine=engine,
        dump_path=db_path,
        query=f"SELECT * FROM {list(table_names)[0]}",
    )

    # create agents with defaults
    agent_config = AgentConfig(
        persona=personas.DEFAULT,
        human=humans.DEFAULT,
        model=DEFAULT_MEMGPT_MODEL,
        data_source="tmp_hf_dataset",
    )

    # create state manager based off loaded data
    persistence_manager = LocalStateManager(agent_config=agent_config)

    # create agent
    memgpt_agent = presets.use_preset(
        presets.DEFAULT,
        agent_config,
        DEFAULT_MEMGPT_MODEL,
        personas.get_persona_text(personas.DEFAULT),
        humans.get_human_text(humans.DEFAULT),
        memgpt.interface,
        persistence_manager,
    )
    print("Successfully loaded into index")
    assert True
