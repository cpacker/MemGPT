from datetime import datetime
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

def read_in_chunks(file_object, chunk_size):
    while True:
        data = file_object.read(chunk_size)
        if not data:
            break
        yield data

def prepare_archival_index_from_files(glob_pattern, tkns_per_chunk=300, model='gpt-4'):
    encoding = tiktoken.encoding_for_model(model)
    files = glob.glob(glob_pattern)
    archival_database = []
    for file in files:
        timestamp = os.path.getmtime(file)
        formatted_time = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %I:%M:%S %p %Z%z")
        with open(file, 'r') as f:
            lines = [l for l in read_in_chunks(f, tkns_per_chunk*4)]
        chunks = [] 
        curr_chunk = []
        curr_token_ct = 0
        for line in lines:
            line = line.rstrip()
            line = line.lstrip()
            try:
                line_token_ct = len(encoding.encode(line))
            except Exception as e:
                line_token_ct = len(line.split(' ')) / .75
                print(f"Could not encode line {line}, estimating it to be {line_token_ct} tokens") 
            if line_token_ct > tkns_per_chunk:
                if len(curr_chunk) > 0:
                    chunks.append(''.join(curr_chunk))
                    curr_chunk = []
                    curr_token_ct = 0
                chunks.append(line[:3200])
                continue
            curr_token_ct += line_token_ct
            curr_chunk.append(line)
            if curr_token_ct > tkns_per_chunk:
                chunks.append(''.join(curr_chunk))
                curr_chunk = []
                curr_token_ct = 0

        if len(curr_chunk) > 0:
            chunks.append(''.join(curr_chunk))

        file_stem = file.split('/')[-1]
        for i, chunk in enumerate(chunks):
            archival_database.append({
                'content': f"[File: {file_stem} Part {i}/{len(chunks)}] {chunk}",
                'timestamp': formatted_time,
            })
    return archival_database

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