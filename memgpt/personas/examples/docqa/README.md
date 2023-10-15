# MemGPT Search over LlamaIndex API Docs

1. 
    a. Download embeddings and docs index from XYZ.
    -- OR --
    b. Build the index:
    1. Build llama_index API docs with `make text`. Instructions [here](https://github.com/run-llama/llama_index/blob/main/docs/DOCS_README.md). Copy over the generated `_build/text` folder to this directory.
    2. Generate embeddings and FAISS index.
        ```bash
        python3 scrape_docs.py
        python3 generate_embeddings_for_docs.py all_docs.jsonl
        python3 build_index.py --embedding_files all_docs.embeddings.jsonl --output_index_file all_docs.index
        ```