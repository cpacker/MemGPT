"""
Script to convert MQuAKE data to letta form.


"""
import argparse
from collections import defaultdict
from openai import OpenAI
import jsonlines
import json

import tqdm

def question_to_statement(question: str) -> str:
    """
    Given a question, return a cloze statement with a blank for the answer.
    """
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": f"""Turn this question into a factual cloze statement with {{}} for where the answer should go, do not actually include the answer in the braces: \n\nQuestion: {question}\n\n Respond with just the statement only."""
            }
        ]
    )
    return completion.choices[0].message.content
    
client = OpenAI()

def convert_explicit_statements(input_file_name: str, output_file_name: str, num_questions: int):
    """
    Convert the MQuAKE data to the Letta format.
    """
    with open(input_file_name) as input_file:
        with jsonlines.open(output_file_name, 'w') as output_file:
            data = json.load(input_file)
            for idx, datum in tqdm.tqdm(enumerate(data[:num_questions])):
                letta_data = defaultdict(list)
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
                    statement = question_to_statement(requested_rewrite)
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
    argparse.add_argument("--num_questions", type=int, required=True, default=100)
    args = argparse.parse_args()
    convert_explicit_statements(args.input_file, args.output_file, args.num_questions)
