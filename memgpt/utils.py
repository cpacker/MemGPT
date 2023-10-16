from datetime import datetime
import difflib
import demjson3 as demjson
import numpy as np
import json
import pytz
import os
import faiss


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
    return ''.join(diff)

def get_local_time_military():
    # Get the current time in UTC
    current_time_utc = datetime.now(pytz.utc)

    # Convert to San Francisco's time zone (PST/PDT)
    sf_time_zone = pytz.timezone('America/Los_Angeles')
    local_time = current_time_utc.astimezone(sf_time_zone)

    # You may format it as you desire
    formatted_time = local_time.strftime("%Y-%m-%d %H:%M:%S %Z%z")

    return formatted_time

def get_local_time():
    # Get the current time in UTC
    current_time_utc = datetime.now(pytz.utc)

    # Convert to San Francisco's time zone (PST/PDT)
    sf_time_zone = pytz.timezone('America/Los_Angeles')
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
    with open(archival_database_file, 'rt') as f:
        all_data = [json.loads(line) for line in f]
    for doc in all_data:
        total = len(doc)
        for i, passage in enumerate(doc):
            archival_database.append({
                'content': f"[Title: {passage['title']}, {i}/{total}] {passage['text']}",
                'timestamp': get_local_time(),
            })  
    return index, archival_database