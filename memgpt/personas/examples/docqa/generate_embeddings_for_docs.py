import asyncio
import json
import os
import logging
import sys
import argparse

from tqdm import tqdm
import openai

try:
    from dotenv import load_dotenv

    load_dotenv()
except ModuleNotFoundError:
    pass
openai.api_key = os.getenv("OPENAI_API_KEY")

sys.path.append("../../../")
from openai_tools import async_get_embedding_with_backoff
from openai_parallel_request_processor import process_api_requests_from_file


# some settings specific to our own OpenAI org limits
# (specific to text-embedding-ada-002)
TPM_LIMIT = 1000000
RPM_LIMIT = 3000

DEFAULT_FILE = "iclr/data/qa_data/30_total_documents/nq-open-30_total_documents_gold_at_0.jsonl.gz"
EMBEDDING_MODEL = "text-embedding-ada-002"


async def generate_requests_file(filename):
    """Generate a file of requests, which we can feed to a pre-made openai cookbook function"""
    base_name = os.path.splitext(filename)[0]
    requests_filename = f"{base_name}_embedding_requests.jsonl"

    with open(filename, "r") as f:
        all_data = [json.loads(line) for line in f]

    with open(requests_filename, "w") as f:
        for data in all_data:
            documents = data
            for idx, doc in enumerate(documents):
                title = doc["title"]
                text = doc["text"]
                document_string = f"Document [{idx+1}] (Title: {title}) {text}"
                request = {"model": EMBEDDING_MODEL, "input": document_string}
                json_string = json.dumps(request)
                f.write(json_string + "\n")

    # Run your parallel processing function
    input(f"Generated requests file ({requests_filename}), continue with embedding batch requests? (hit enter)")
    await process_api_requests_from_file(
        requests_filepath=requests_filename,
        save_filepath=f"{base_name}.embeddings.jsonl.gz",  # Adjust as necessary
        request_url="https://api.openai.com/v1/embeddings",
        api_key=os.getenv("OPENAI_API_KEY"),
        max_requests_per_minute=RPM_LIMIT,
        max_tokens_per_minute=TPM_LIMIT,
        token_encoding_name=EMBEDDING_MODEL,
        max_attempts=5,
        logging_level=logging.INFO,
    )


async def generate_embedding_file(filename, parallel_mode=False):
    if parallel_mode:
        await generate_requests_file(filename)
        return

    # Derive the sister filename
    # base_name = os.path.splitext(filename)[0]
    base_name = filename.rsplit(".jsonl", 1)[0]
    sister_filename = f"{base_name}.embeddings.jsonl"

    # Check if the sister file already exists
    if os.path.exists(sister_filename):
        print(f"{sister_filename} already exists. Skipping embedding generation.")
        return

    with open(filename, "rt") as f:
        all_data = [json.loads(line) for line in f]

    embedding_data = []
    total_documents = sum(len(data) for data in all_data)

    # Outer loop progress bar
    for i, data in enumerate(tqdm(all_data, desc="Processing data", total=len(all_data))):
        documents = data
        # Inner loop progress bar
        for idx, doc in enumerate(
            tqdm(documents, desc=f"Embedding documents for data {i+1}/{len(all_data)}", total=len(documents), leave=False)
        ):
            title = doc["title"]
            text = doc["text"]
            document_string = f"[Title: {title}] {text}"
            try:
                embedding = await async_get_embedding_with_backoff(document_string, model=EMBEDDING_MODEL)
            except Exception as e:
                print(document_string)
                raise e
            embedding_data.append(embedding)

    # Save the embeddings to the sister file
    # with gzip.open(sister_filename, 'wt') as f:
    with open(sister_filename, "wb") as f:
        for embedding in embedding_data:
            # f.write(json.dumps(embedding) + '\n')
            f.write((json.dumps(embedding) + "\n").encode("utf-8"))

    print(f"Embeddings saved to {sister_filename}")


async def main():
    if len(sys.argv) > 1:
        filename = sys.argv[1]
    else:
        filename = DEFAULT_FILE
    await generate_embedding_file(filename)


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", nargs="?", default=DEFAULT_FILE, help="Path to the input file")
    parser.add_argument("--parallel", action="store_true", help="Enable parallel mode")
    args = parser.parse_args()

    await generate_embedding_file(args.filename, parallel_mode=args.parallel)


if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
