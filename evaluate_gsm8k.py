"""
Evaluate the results for the GSM8K dataset 

Usage:

    python evaluate_gsm8k.py --input_file <path_to_input_file>
"""
import argparse
import jsonlines
import json
import re

def evaluate(input_file: str):
    correct = 0
    total = 0
    with jsonlines.open(input_file) as reader:
        for obj in reader:
            ignore_regex = '(?s).*#### '
            answer = re.sub(ignore_regex, '', obj['answer'])
            final_answer = ""
            for message in obj['response']['messages']:
                if message['message_type'] == "function_call":
                    if message['function_call']['name'] == "send_message":
                        arguments = json.loads(message['function_call']['arguments'])
                        final_answer = arguments['message']

            regex_str = "(-?[$0-9.,]{2,})|(-?[0-9]+)"
            matches = re.findall(regex_str, final_answer)
            final_num = "".join(matches[-1])

            if float(final_num) == float(answer):
                correct += 1
            total += 1

    
    print("Accuracy: ", correct / total)

    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    args = parser.parse_args()

    evaluate(args.input_file)