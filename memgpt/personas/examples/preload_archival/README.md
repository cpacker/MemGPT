# Preloading Archival Memory with Files
MemGPT enables you to chat with your data locally -- this example gives the workflow for loading documents into MemGPT's archival memory.

To run our example where you can search over the SEC 10-K filings of Uber, Lyft, and Airbnb,

1. Download the .txt files from [Hugging Face](https://huggingface.co/datasets/MemGPT/example-sec-filings/tree/main) and place them in this directory.

2. In the root `MemGPT` directory, run
    ```bash
    python3 main.py --archival_storage_files="memgpt/personas/examples/preload_archival/*.txt" --persona=memgpt_doc --human=basic
    ```


If you would like to load your own local files into MemGPT's archival memory, run the command above but replace `--archival_storage_files="memgpt/personas/examples/preload_archival/*.txt"` with your own file glob expression (enclosed in quotes).

## Demo
<div align="center">
    <img src="https://memgpt.ai/assets/img/preload_archival_demo.gif" alt="MemGPT demo video for searching through preloaded files" width="800">
</div>
