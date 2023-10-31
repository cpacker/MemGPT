import faiss
from glob import glob
from tqdm import tqdm
import numpy as np
import argparse
import json


def build_index(embedding_files: str, index_name: str):
    index = faiss.IndexFlatL2(1536)
    file_list = sorted(glob(embedding_files))

    for embedding_file in file_list:
        print(embedding_file)
        with open(embedding_file, "rt", encoding="utf-8") as file:
            embeddings = []
            l = 0
            for line in tqdm(file):
                # Parse each JSON line
                data = json.loads(line)
                embeddings.append(data)
                l += 1
            data = np.array(embeddings).astype("float32")
            print(data.shape)
            try:
                index.add(data)
            except Exception as e:
                print(data)
                raise e

    faiss.write_index(index, index_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--embedding_files", type=str, help="embedding_filepaths glob expression")
    parser.add_argument("--output_index_file", type=str, help="output filepath")
    args = parser.parse_args()

    build_index(embedding_files=args.embedding_files, index_name=args.output_index_file)
