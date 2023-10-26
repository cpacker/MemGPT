# MemGPT over LlamaIndex API Docs

MemGPT enables you to chat with your data -- try running this example to talk to the LlamaIndex API docs!

1.
    a. Download LlamaIndex API docs and FAISS index from [Hugging Face](https://huggingface.co/datasets/MemGPT/llamaindex-api-docs).
   ```bash
   # Make sure you have git-lfs installed (https://git-lfs.com)
   git lfs install
   git clone https://huggingface.co/datasets/MemGPT/llamaindex-api-docs
   ```

    **-- OR --**

    b. Build the index:
    1. Build `llama_index` API docs with `make text`. Instructions [here](https://github.com/run-llama/llama_index/blob/main/docs/DOCS_README.md). Copy over the generated `_build/text` folder to this directory.
    2. Generate embeddings and FAISS index.
        ```bash
        python3 scrape_docs.py
        python3 generate_embeddings_for_docs.py all_docs.jsonl
        python3 build_index.py --embedding_files all_docs.embeddings.jsonl --output_index_file all_docs.index
        ```

2. In the root `MemGPT` directory, run
    ```bash
    python3 main.py --archival_storage_faiss_path=<ARCHIVAL_STORAGE_FAISS_PATH> --persona=memgpt_doc --human=basic
    ```
    where `ARCHIVAL_STORAGE_FAISS_PATH` is the directory where `all_docs.jsonl` and `all_docs.index` are located.
   If you downloaded from Hugging Face, it will be `memgpt/personas/docqa/llamaindex-api-docs`.
   If you built the index yourself, it will be `memgpt/personas/docqa`.

## Demo
<div align="center">
    <img src="https://memgpt.ai/assets/img/docqa_demo.gif" alt="MemGPT demo video for llamaindex api docs search" width="800">
</div>
