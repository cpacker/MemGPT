'''
Script to generate GSM8k-Memory dataset.

Example:
    python fact_editing_experiments/generate_gsm8k_memory.py --output_filename gsm8k_memory.jsonl --k 10

'''
import argparse
import random
from flask import g
import jsonlines
from datasets import load_dataset

ds = load_dataset("openai/gsm8k", "main")


def generate_dataset(output_filename: str, k: int):
    with jsonlines.open(output_filename, mode="a") as writer:
        
        gsm8k_memory_instances = []
        for test_instance in ds["test"]:
            try:
                gsm8k_memory_instance = {}
                sentences = test_instance["question"].rsplit('.', 1)
                gsm8k_memory_instance["context"] = sentences[0].strip() + "."
                gsm8k_memory_instance["question"] = sentences[1].strip()

                # sample randoml k instances from train dataset
                sampled_instances = random.sample(list(ds["train"]), k)

                gsm8k_memory_instance["memory"] = [sampled_instance["question"] for sampled_instance in sampled_instances]
                gsm8k_memory_instance["answer"] = test_instance["answer"] 

                gsm8k_memory_instance["metadata"] = {"train_instances": sampled_instances}
                gsm8k_memory_instances.append(gsm8k_memory_instance)
            except:
                print("Could not split", test_instance)
                continue
        writer.write_all(gsm8k_memory_instances)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate GSM8k-Memory predictions")
    parser.add_argument("--output_filename", type=str, help="Path to the predictions file")
    parser.add_argument("--k", type=int, help="number of dataset instances to sample")
    args = parser.parse_args()
    generate_dataset(args.output_filename, args.k)
