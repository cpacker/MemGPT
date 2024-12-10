"""
Script to convert MQuAKE data to letta form.

Example:
    
    python convert_mquake_to_explicit_format.py --input_file MQuaAKE/datasets/MQuAKE-CF-3k-v2.json --output_file MQuAKE/datasets/letta-MQuAKE-CF-3k-v2.json --num_questions 1000

"""
import argparse
from collections import defaultdict
from openai import OpenAI
import jsonlines
import json
from typing import Optional
import tqdm

def question_to_statement(question: str) -> str:
    """
    Given a question, return a cloze statement with a blank for the answer. 
    We use this to convert the questions into a chat message to the model
    """
    completion = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": f"""Turn this question into a factual cloze statement with {{}} for where the answer should go, do not actually include the answer in the braces or any spaces between the braces. This is to be used for formatting.: \n\nQuestion: {question}\n\n Respond with just the statement only."""
            }
        ]
    )
    return completion.choices[0].message.content
    
client = OpenAI()

def convert_explicit_statements(input_file_name: str, output_file_name: str, num_questions: Optional[int]):
    """
    Convert the MQuAKE data to the Letta format.
    """
    with open(input_file_name) as input_file:
        with jsonlines.open(output_file_name, 'w') as output_file:
            data = json.load(input_file)
            if num_questions is None:
                data = data[:num_questions]
            for idx, datum in tqdm.tqdm(enumerate(data)):
                letta_data = defaultdict(list)

                # Add metadata
                for key in ["case_id", 
                            "answer", "answer_alias",
                            "new_answer", "new_answer_alias",
                            "orig"]:
                    letta_data[key] = datum[key]

                for hop in datum["single_hops"]:
                    fact = f"{hop["cloze"]} {hop["answer"]}."
                    letta_data['memory'].append(fact)

                for hop in datum["new_single_hops"]:
                    fact = f"{hop["cloze"]} {hop["answer"]}."
                    letta_data['new_memory'].append(fact)

                for question in datum["questions"]: 
                    statement = question_to_statement(question)
                    letta_data['multihop_statements'].append(statement)

                for hop in letta_data['multihop_statements']:
                    if hop.count("{}") == 1:
                        letta_data['memory_multi_hop'].append(hop.format(datum["answer"]))
                    else:
                        print("Multiple role bindings in the statement: ", hop)
                        pass
                
                for hop in letta_data['multihop_statements']:
                    if hop.count("{}") == 1:
                        letta_data['new_memory_multi_hop'].append(hop.format(datum["new_answer"]))
                    else:
                        print("Multiple role bindings in the statement", hop)
                        pass

                for requested_rewrite in datum["requested_rewrite"]:
                    letta_data['requested_rewrites'].append(
                        requested_rewrite["prompt"].format(requested_rewrite["subject"]) + " " + requested_rewrite["target_new"]["str"]
                        )

                output_file.write(letta_data)

                if idx == num_questions:
                    break

            
if __name__ == "__main__":
    argparse = argparse.ArgumentParser()
    argparse.add_argument("--input_file", type=str, required=True, default="MQuaAKE/datasets/MQuAKE-CF-3k-v2.json")
    argparse.add_argument("--output_file", type=str, required=True, default="MQuAKE/datasets/letta-MQuAKE-CF-3k-v2.json")
    argparse.add_argument("--num_questions", type=int, required=False, default=None)
    args = argparse.parse_args()
    convert_explicit_statements(args.input_file, args.output_file, args.num_questions)
