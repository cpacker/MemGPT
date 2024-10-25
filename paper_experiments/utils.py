import gzip
import json
from typing import List

from letta.config import LettaConfig


def load_gzipped_file(file_path):
    with gzip.open(file_path, "rt", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def read_jsonl(filename) -> List[dict]:
    lines = []
    with open(filename, "r") as file:
        for line in file:
            lines.append(json.loads(line.strip()))
    return lines


def get_experiment_config(postgres_uri, endpoint_type="openai", model="gpt-4"):
    config = LettaConfig.load()
    config.archival_storage_type = "postgres"
    config.archival_storage_uri = postgres_uri

    config = LettaConfig(
        archival_storage_type="postgres",
        archival_storage_uri=postgres_uri,
        recall_storage_type="postgres",
        recall_storage_uri=postgres_uri,
        metadata_storage_type="postgres",
        metadata_storage_uri=postgres_uri,
    )
    return config
