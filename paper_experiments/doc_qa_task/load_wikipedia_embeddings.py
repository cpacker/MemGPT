import copy
import hashlib
import json
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed

from absl import app, flags
from icml_experiments.utils import get_experiment_config
from tqdm import tqdm

from letta.agent_store.storage import StorageConnector, TableType
from letta.cli.cli_config import delete
from letta.data_types import Passage

# Create an empty list to store the JSON objects
source_name = "wikipedia"
config = get_experiment_config(os.environ.get("PGVECTOR_TEST_DB_URL"), endpoint_type="openai")
config.save()  # save config to file
user_id = uuid.UUID(config.anon_clientid)

FLAGS = flags.FLAGS
flags.DEFINE_boolean("drop_db", default=False, required=False, help="Drop existing source DB")
flags.DEFINE_string("file", default=None, required=True, help="File to parse")


def create_uuid_from_string(val: str):
    """
    Generate consistent UUID from a string
    from: https://samos-it.com/posts/python-create-uuid-from-random-string-of-words.html
    """
    hex_string = hashlib.md5(val.encode("UTF-8")).hexdigest()
    return uuid.UUID(hex=hex_string)


def insert_lines(lines, conn, show_progress=False):
    """Parse and insert list of lines into source database"""
    passages = []
    iterator = tqdm(lines) if show_progress else lines
    added = set()
    for line in iterator:
        d = json.loads(line)
        # pprint(d)
        assert len(d) == 2, f"Line is empty: {len(d)}"
        text = d[0]["input"]
        model = d[0]["model"]
        embedding = d[1]["data"][0]["embedding"]
        embedding_dim = len(embedding)
        assert embedding_dim == 1536, f"Wrong embedding dim: {len(embedding_dim)}"
        assert len(d[1]["data"]) == 1, f"More than one embedding: {len(d[1]['data'])}"
        d[1]["usage"]
        # print(text)

        passage_id = create_uuid_from_string(text)  # consistent hash for text (prevent duplicates)
        if passage_id in added:
            continue
        else:
            added.add(passage_id)
        # if conn.get(passage_id):
        #    continue

        passage = Passage(
            id=passage_id,
            user_id=user_id,
            text=text,
            embedding_model=model,
            embedding_dim=embedding_dim,
            embedding=embedding,
            # metadata=None,
            data_source=source_name,
        )
        # print(passage.id)
        passages.append(passage)
    st = time.time()
    # insert_passages_into_source(passages, source_name=source_name, user_id=user_id, config=config)
    # conn.insert_many(passages)
    conn.upsert_many(passages)
    return time.time() - st


def main(argv):
    # clear out existing source
    if FLAGS.drop_db:
        delete("source", source_name)
        try:
            passages_table = StorageConnector.get_storage_connector(TableType.PASSAGES, config, user_id)
            passages_table.delete_table()

        except Exception as e:
            print("Failed to delete source")
            print(e)

    # Open the file and read line by line
    count = 0
    # files = [
    #    #'data/wikipedia_passages_shard_1-00.jsonl',
    #    #'data/wikipedia_passages_shard_1-01.jsonl',
    #    'data/wikipedia_passages_shard_1-02.jsonl',
    #    #'data/wikipedia_passages_shard_1-03.jsonl',
    #    #'data/wikipedia_passages_shard_1-04.jsonl',
    #    #'data/wikipedia_passages_shard_1-05.jsonl',
    #    #'data/wikipedia_passages_shard_1-06.jsonl',
    #    #'data/wikipedia_passages_shard_1-07.jsonl',
    #    #'data/wikipedia_passages_shard_1-08.jsonl',
    #    #'data/wikipedia_passages_shard_1-09.jsonl',
    # ]
    files = [FLAGS.file]
    chunk_size = 1000
    conn = StorageConnector.get_storage_connector(TableType.PASSAGES, config, user_id)
    for file_path in files:
        print(file_path)
        futures = []
        with ThreadPoolExecutor(max_workers=64) as p:
            with open(file_path, "r") as file:
                lines = []

                # insert lines in 1k chunks
                for line in tqdm(file):
                    lines.append(line)
                    if len(lines) >= chunk_size:
                        if count == 0:
                            # future = p.submit(insert_lines, copy.deepcopy(lines), conn, True)
                            print("Await first result (hack to avoid concurrency issues)")
                            t = insert_lines(lines, conn, True)
                            # t = future.result()
                            print("Finished first result", t)
                        else:
                            future = p.submit(insert_lines, copy.deepcopy(lines), conn)
                            futures.append(future)
                        count += len(lines)
                        lines = []

                # insert remaining lines
                if len(lines) > 0:
                    future = p.submit(insert_lines, copy.deepcopy(lines), conn)
                    futures.append(future)
                    count += len(lines)
                    lines = []

                    ## breaking point
                    # if count >= 3000:
                    #    break

            print(f"Waiting for {len(futures)} futures")
            # wait for futures
            for future in tqdm(as_completed(futures)):
                future.result()

        # check metadata
        # storage = StorageConnector.get_storage_connector(TableType.PASSAGES, config, user_id)
        # size = storage.size()
        size = conn.size()
        print("Number of passages", size)


if __name__ == "__main__":
    app.run(main)
