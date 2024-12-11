'''
Script to generate GSM8k-Memory dataset.

Example:
    python fact_editing_experiments/generate_gsm8k_memory.py --output_filename gsm8k_memory.jsonl --k 10

'''
import argparse
import random
import jsonlines
from datasets import load_dataset

ds = load_dataset("openai/gsm8k", "main")


def generate_dataset(output_filename: str, k: int):
    with jsonlines.open(output_filename, mode="w") as writer:
        for test_instance in ds["test"]:
            gsm8k_memory_instances = {}
            sentences = test_instance["question"].rsplit('.', 1)
            gsm8k_memory_instances["context"] = sentences[0].strip() + "."
            gsm8k_memory_instances["question"] = sentences[1].strip()

            # sample randoml k instances from train dataset
            sampled_instances = random.sample(list(ds["train"]), k)

            gsm8k_memory_instances["memory"] = [sampled_instances["question"] for sampled_instances in sampled_instances]
            gsm8k_memory_instances["answer"] = test_instance["answer"] 

            gsm8k_memory_instances["metadata"] = {"train_instances": sampled_instances}
            
            writer.write(gsm8k_memory_instances)
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate GSM8k-Memory predictions")
    parser.add_argument("--output_filename", type=str, help="Path to the predictions file")
    parser.add_argument("--k", type=int, help="number of dataset instances to sample")
    args = parser.parse_args()
    generate_dataset(args.output_filename, args.k)
