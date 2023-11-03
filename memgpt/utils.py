from datetime import datetime
import asyncio
import csv
import difflib
import demjson3 as demjson
import numpy as np
import json
import pytz
import os
import faiss
import tiktoken
import glob
import sqlite3
import fitz
from tqdm import tqdm
import typer
import memgpt
from memgpt.openai_tools import async_get_embedding_with_backoff
from memgpt.constants import MEMGPT_DIR
from llama_index import set_global_service_context, ServiceContext, VectorStoreIndex, load_index_from_storage, StorageContext
from llama_index.embeddings import OpenAIEmbedding


def count_tokens(s: str, model: str = "gpt-4") -> int:
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(s))


# DEBUG = True
DEBUG = False


def printd(*args, **kwargs):
    if DEBUG:
        print(*args, **kwargs)


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def united_diff(str1, str2):
    lines1 = str1.splitlines(True)
    lines2 = str2.splitlines(True)
    diff = difflib.unified_diff(lines1, lines2)
    return "".join(diff)


def get_local_time_military():
    # Get the current time in UTC
    current_time_utc = datetime.now(pytz.utc)

    # Convert to San Francisco's time zone (PST/PDT)
    sf_time_zone = pytz.timezone("America/Los_Angeles")
    local_time = current_time_utc.astimezone(sf_time_zone)

    # You may format it as you desire
    formatted_time = local_time.strftime("%Y-%m-%d %H:%M:%S %Z%z")

    return formatted_time


def get_local_time():
    # Get the current time in UTC
    current_time_utc = datetime.now(pytz.utc)

    # Convert to San Francisco's time zone (PST/PDT)
    sf_time_zone = pytz.timezone("America/Los_Angeles")
    local_time = current_time_utc.astimezone(sf_time_zone)

    # You may format it as you desire, including AM/PM
    formatted_time = local_time.strftime("%Y-%m-%d %I:%M:%S %p %Z%z")

    return formatted_time


def parse_json(string):
    result = None
    try:
        result = json.loads(string)
        return result
    except Exception as e:
        print(f"Error parsing json with json package: {e}")

    try:
        result = demjson.decode(string)
        return result
    except demjson.JSONDecodeError as e:
        print(f"Error parsing json with demjson package: {e}")
        raise e


def prepare_archival_index(folder):
    index_file = os.path.join(folder, "all_docs.index")
    index = faiss.read_index(index_file)

    archival_database_file = os.path.join(folder, "all_docs.jsonl")
    archival_database = []
    with open(archival_database_file, "rt") as f:
        all_data = [json.loads(line) for line in f]
    for doc in all_data:
        total = len(doc)
        for i, passage in enumerate(doc):
            archival_database.append(
                {
                    "content": f"[Title: {passage['title']}, {i}/{total}] {passage['text']}",
                    "timestamp": get_local_time(),
                }
            )
    return index, archival_database


def read_in_chunks(file_object, chunk_size):
    while True:
        data = file_object.read(chunk_size)
        if not data:
            break
        yield data


def read_pdf_in_chunks(file, chunk_size):
    doc = fitz.open(file)
    for page in doc:
        text = page.get_text()
        yield text


def read_in_rows_csv(file_object, chunk_size):
    csvreader = csv.reader(file_object)
    header = next(csvreader)
    for row in csvreader:
        next_row_terms = []
        for h, v in zip(header, row):
            next_row_terms.append(f"{h}={v}")
        next_row_str = ", ".join(next_row_terms)
        yield next_row_str


def prepare_archival_index_from_files(glob_pattern, tkns_per_chunk=300, model="gpt-4"):
    encoding = tiktoken.encoding_for_model(model)
    files = glob.glob(glob_pattern, recursive=True)
    return chunk_files(files, tkns_per_chunk, model)


def total_bytes(pattern):
    total = 0
    for filename in glob.glob(pattern, recursive=True):
        if os.path.isfile(filename):  # ensure it's a file and not a directory
            total += os.path.getsize(filename)
    return total


def chunk_file(file, tkns_per_chunk=300, model="gpt-4"):
    encoding = tiktoken.encoding_for_model(model)

    if file.endswith(".db"):
        return  # can't read the sqlite db this way, will get handled in main.py

    with open(file, "r") as f:
        if file.endswith(".pdf"):
            lines = [l for l in read_pdf_in_chunks(file, tkns_per_chunk * 8)]
            if len(lines) == 0:
                print(f"Warning: {file} did not have any extractable text.")
        elif file.endswith(".csv"):
            lines = [l for l in read_in_rows_csv(f, tkns_per_chunk * 8)]
        else:
            lines = [l for l in read_in_chunks(f, tkns_per_chunk * 4)]
    curr_chunk = []
    curr_token_ct = 0
    for i, line in enumerate(lines):
        line = line.rstrip()
        line = line.lstrip()
        line += "\n"
        try:
            line_token_ct = len(encoding.encode(line))
        except Exception as e:
            line_token_ct = len(line.split(" ")) / 0.75
            print(f"Could not encode line {i}, estimating it to be {line_token_ct} tokens")
            print(e)
        if line_token_ct > tkns_per_chunk:
            if len(curr_chunk) > 0:
                yield "".join(curr_chunk)
                curr_chunk = []
                curr_token_ct = 0
            yield line[:3200]
            continue
        curr_token_ct += line_token_ct
        curr_chunk.append(line)
        if curr_token_ct > tkns_per_chunk:
            yield "".join(curr_chunk)
            curr_chunk = []
            curr_token_ct = 0

    if len(curr_chunk) > 0:
        yield "".join(curr_chunk)


def chunk_files(files, tkns_per_chunk=300, model="gpt-4"):
    archival_database = []
    for file in files:
        timestamp = os.path.getmtime(file)
        formatted_time = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %I:%M:%S %p %Z%z")
        file_stem = file.split("/")[-1]
        chunks = [c for c in chunk_file(file, tkns_per_chunk, model)]
        for i, chunk in enumerate(chunks):
            archival_database.append(
                {
                    "content": f"[File: {file_stem} Part {i}/{len(chunks)}] {chunk}",
                    "timestamp": formatted_time,
                }
            )
    return archival_database


def chunk_files_for_jsonl(files, tkns_per_chunk=300, model="gpt-4"):
    ret = []
    for file in files:
        file_stem = file.split("/")[-1]
        curr_file = []
        for chunk in chunk_file(file, tkns_per_chunk, model):
            curr_file.append(
                {
                    "title": file_stem,
                    "text": chunk,
                }
            )
        ret.append(curr_file)
    return ret


async def process_chunk(i, chunk, model):
    try:
        return i, await async_get_embedding_with_backoff(chunk["content"], model=model)
    except Exception as e:
        print(chunk)
        raise e


async def process_concurrently(archival_database, model, concurrency=10):
    # Create a semaphore to limit the number of concurrent tasks
    semaphore = asyncio.Semaphore(concurrency)

    async def bounded_process_chunk(i, chunk):
        async with semaphore:
            return await process_chunk(i, chunk, model)

    # Create a list of tasks for chunks
    embedding_data = [0 for _ in archival_database]
    tasks = [bounded_process_chunk(i, chunk) for i, chunk in enumerate(archival_database)]

    for future in tqdm(
        asyncio.as_completed(tasks),
        total=len(archival_database),
        desc="Processing file chunks",
    ):
        i, result = await future
        embedding_data[i] = result

    return embedding_data


async def prepare_archival_index_from_files_compute_embeddings(
    glob_pattern,
    tkns_per_chunk=300,
    model="gpt-4",
    embeddings_model="text-embedding-ada-002",
):
    files = sorted(glob.glob(glob_pattern, recursive=True))
    save_dir = os.path.join(
        MEMGPT_DIR,
        "archival_index_from_files_" + get_local_time().replace(" ", "_").replace(":", "_"),
    )
    os.makedirs(save_dir, exist_ok=True)
    total_tokens = total_bytes(glob_pattern) / 3
    price_estimate = total_tokens / 1000 * 0.0001
    confirm = input(f"Computing embeddings over {len(files)} files. This will cost ~${price_estimate:.2f}. Continue? [y/n] ")
    if confirm != "y":
        raise Exception("embeddings were not computed")

    # chunk the files, make embeddings
    archival_database = chunk_files(files, tkns_per_chunk, model)
    embedding_data = await process_concurrently(archival_database, embeddings_model)
    embeddings_file = os.path.join(save_dir, "embeddings.json")
    with open(embeddings_file, "w") as f:
        print(f"Saving embeddings to {embeddings_file}")
        json.dump(embedding_data, f)

    # make all_text.json
    archival_storage_file = os.path.join(save_dir, "all_docs.jsonl")
    chunks_by_file = chunk_files_for_jsonl(files, tkns_per_chunk, model)
    with open(archival_storage_file, "w") as f:
        print(f"Saving archival storage with preloaded files to {archival_storage_file}")
        for c in chunks_by_file:
            json.dump(c, f)
            f.write("\n")

    # make the faiss index
    index = faiss.IndexFlatL2(1536)
    data = np.array(embedding_data).astype("float32")
    try:
        index.add(data)
    except Exception as e:
        print(data)
        raise e
    index_file = os.path.join(save_dir, "all_docs.index")
    print(f"Saving faiss index {index_file}")
    faiss.write_index(index, index_file)
    return save_dir


def read_database_as_list(database_name):
    result_list = []

    try:
        conn = sqlite3.connect(database_name)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        table_names = cursor.fetchall()
        for table_name in table_names:
            cursor.execute(f"PRAGMA table_info({table_name[0]});")
            schema_rows = cursor.fetchall()
            columns = [row[1] for row in schema_rows]
            cursor.execute(f"SELECT * FROM {table_name[0]};")
            rows = cursor.fetchall()
            result_list.append(f"Table: {table_name[0]}")  # Add table name to the list
            schema_row = "\t".join(columns)
            result_list.append(schema_row)
            for row in rows:
                data_row = "\t".join(map(str, row))
                result_list.append(data_row)
        conn.close()
    except sqlite3.Error as e:
        result_list.append(f"Error reading database: {str(e)}")
    except Exception as e:
        result_list.append(f"Error: {str(e)}")
    return result_list


def estimate_openai_cost(docs):
    """Estimate OpenAI embedding cost

    :param docs: Documents to be embedded
    :type docs: List[Document]
    :return: Estimated cost
    :rtype: float
    """
    from llama_index import MockEmbedding
    from llama_index.callbacks import CallbackManager, TokenCountingHandler
    import tiktoken

    embed_model = MockEmbedding(embed_dim=1536)

    token_counter = TokenCountingHandler(tokenizer=tiktoken.encoding_for_model("gpt-3.5-turbo").encode)

    callback_manager = CallbackManager([token_counter])

    set_global_service_context(ServiceContext.from_defaults(embed_model=embed_model, callback_manager=callback_manager))
    index = VectorStoreIndex.from_documents(docs)

    # estimate cost
    cost = 0.0001 * token_counter.total_embedding_token_count / 1000
    token_counter.reset_counts()
    return cost


def get_index(name, docs):
    """Index documents

    :param docs: Documents to be embedded
    :type docs: List[Document]
    """
    from memgpt.config import MemGPTConfig  # avoid circular import
    from memgpt.embeddings import embedding_model  # avoid circular import

    # TODO: configure to work for local
    print("Warning: get_index(docs) only supported for OpenAI")

    # check if directory exists
    dir = f"{MEMGPT_DIR}/archival/{name}"
    if os.path.exists(dir):
        confirm = typer.confirm(typer.style(f"Index with name {name} already exists -- re-index?", fg="yellow"), default=False)
        if not confirm:
            # return existing index
            storage_context = StorageContext.from_defaults(persist_dir=dir)
            return load_index_from_storage(storage_context)

    # TODO: support configurable embeddings
    # TODO: read from config how to index (open ai vs. local): then embed_mode="local"

    estimated_cost = estimate_openai_cost(docs)
    # TODO: prettier cost formatting
    confirm = typer.confirm(
        typer.style(f"Open AI embedding cost will be approximately ${estimated_cost} - continue?", fg="yellow"), default=True
    )

    if not confirm:
        typer.secho("Aborting.", fg="red")
        exit()

    # read embedding confirguration
    # TODO: in the future, make an IngestData class that loads the config once
    config = MemGPTConfig.load()
    embed_model = embedding_model(config)
    chunk_size = config.embedding_chunk_size
    service_context = ServiceContext.from_defaults(embed_model=embed_model, chunk_size=chunk_size)
    set_global_service_context(service_context)

    # index documents
    index = VectorStoreIndex.from_documents(docs)
    return index


def save_agent_index(index, agent_config):
    """Save agent index inside of ~/.memgpt/agents/<agent_name>

    :param index: Index to save
    :type index: VectorStoreIndex
    :param agent_name: Name of agent that the archival memory belonds to
    :type agent_name: str
    """
    dir = agent_config.save_agent_index_dir()
    os.makedirs(dir, exist_ok=True)
    index.storage_context.persist(dir)


def save_index(index, name):
    """Save index ~/.memgpt/archival/<name> to load into agents

    :param index: Index to save
    :type index: VectorStoreIndex
    :param name: Name of index
    :type name: str
    """
    # save
    # TODO: load directory from config
    # TODO: save to vectordb/local depending on config

    dir = f"{MEMGPT_DIR}/archival/{name}"

    ## Avoid overwriting
    ## check if directory exists
    # if os.path.exists(dir):
    #    confirm = typer.confirm(typer.style(f"Index with name {name} already exists -- overwrite?", fg="red"), default=False)
    #    if not confirm:
    #        typer.secho("Aborting.", fg="red")
    #        exit()

    # create directory, even if it already exists
    os.makedirs(dir, exist_ok=True)
    index.storage_context.persist(dir)
    print(dir)


def list_agent_config_files():
    """List all agents config files"""
    return os.listdir(os.path.join(MEMGPT_DIR, "agents"))


def list_human_files():
    """List all humans files"""
    defaults_dir = os.path.join(memgpt.__path__[0], "humans", "examples")
    user_dir = os.path.join(MEMGPT_DIR, "humans")

    memgpt_defaults = os.listdir(defaults_dir)
    memgpt_defaults = [os.path.join(defaults_dir, f) for f in memgpt_defaults if f.endswith(".txt")]

    user_added = os.listdir(user_dir)
    user_added = [os.path.join(user_dir, f) for f in user_added]
    return memgpt_defaults + user_added


def list_persona_files():
    """List all personas files"""
    defaults_dir = os.path.join(memgpt.__path__[0], "personas", "examples")
    user_dir = os.path.join(MEMGPT_DIR, "personas")

    memgpt_defaults = os.listdir(defaults_dir)
    memgpt_defaults = [os.path.join(defaults_dir, f) for f in memgpt_defaults if f.endswith(".txt")]

    user_added = os.listdir(user_dir)
    user_added = [os.path.join(user_dir, f) for f in user_added]
    return memgpt_defaults + user_added


def get_human_text(name: str):
    for file_path in list_human_files():
        file = os.path.basename(file_path)
        if f"{name}.txt" == file or name == file:
            return open(file_path, "r").read().strip()
    raise ValueError(f"Human {name} not found")


def get_persona_text(name: str):
    for file_path in list_persona_files():
        file = os.path.basename(file_path)
        if f"{name}.txt" == file or name == file:
            return open(file_path, "r").read().strip()

    raise ValueError(f"Persona {name} not found")


def get_human_text(name: str):
    for file_path in list_human_files():
        file = os.path.basename(file_path)
        if f"{name}.txt" == file or name == file:
            return open(file_path, "r").read().strip()
