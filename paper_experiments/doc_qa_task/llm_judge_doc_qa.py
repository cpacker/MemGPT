import argparse
import json
import re

from openai import OpenAI
from tqdm import tqdm

from letta.credentials import LettaCredentials

# Note: did not end up using since no cases of cheating were observed
# CHEATING_PROMPT = \
#    """
#    You are evaluating an LLM trying to do task.
#    The main point of failure is the LLM may cheat and hallucinate the document it is meant to provide as evidence.
#    For example, even though the LLM has been instructed to provide the document that it's using as evidence to answer the question, in this case the LLM hallucinates a document:
#
#    The format of the LLM response with be:
#    ANSWER: <answer>, DOCUMENT: <document_text>
#
#    We also provide the list of documents that the LLM should have used to answer the question.
#    This is provided with the format:
#    ALL DOCUMENTS: <list of documents>
#
#    Your task is to ensure the LLM did not cheat and hallucinate the document.
#    You can ensure this by checking if the document provided by the LLM is in the list of documents provided.
#
#    Write a short reason for your answer before stating the final answer (format <your reason> - CHEAT/CORRECT).
#    """

EVAL_PROMPT = """
    Your task is to evaluate whether an LLM correct answered a question.
    The LLM response should be the format 'ANSWER: <answer>, DOCUMENT: <document_text>' or say 'INSUFFICIENT INFORMATION'.
    The true answer is provided in the format 'TRUE ANSWER: <list of possible answers>'.
    The questions is provided in the format 'QUESTION: <question>'.
    If the LLM response contains both the correct answer and corresponding document text, the response is correct.
    Even if the LLM's answer and the true answer are slightly different in wording, the response is still correct.
    For example, if the answer is more specific than the true answer or uses a different phrasing that is still correct, the response is correct.
    If the LLM response if 'INSUFFICIENT INFORMATION', or the 'DOCUMENT' field is missing, the response is incorrect.
    Respond with a single token: 'CORRECT' or 'INCORRECT'.
    """

EVAL_MODEL = "gpt-4-0613"


def evaluate_response(output: str):
    credentials = LettaCredentials().load()
    assert credentials.openai_key is not None, credentials.openai_key

    client = OpenAI(api_key=credentials.openai_key)

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": "\n".join([EVAL_PROMPT, "\n", output, "\n"]),
            },
        ],
        model=EVAL_MODEL,
    )

    response = chat_completion.choices[0].message.content
    print("llm judge", response)
    if "INCORRECT" in response:
        return False
    elif "CORRECT" in response:
        return True
    else:
        print("INVALID RESPONSE", response)
        return False


# Grab the last thing Letta generated, treat it as the reply
def extract_final_letta_response(letta_responses: list) -> str:
    final_index = -1
    if "function_return" in letta_responses[final_index]:
        final_index = -2
    final_letta_response = [v for k, v in letta_responses[final_index].items()]
    final_letta_response = final_letta_response[-1]
    return final_letta_response


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test script")
    parser.add_argument("--file", type=str, help="File data to evaluate")
    parser.add_argument("--baseline", action="store_true", help="Whether to use the baseline model")
    args = parser.parse_args()

    # load data
    data = json.load(open(args.file))

    # counters
    correct = 0
    total = 0

    # Make an intial pass to determine how many documents had the correct answer
    results = []  # store all results
    eval_results = []  # store results that need LLM judge
    if args.baseline:
        # baseline experiment
        match = re.search(r"model_([^_]+)_num_docs_([^\.]+)\.json", args.file)
        model = match.group(1)
        num_docs = int(match.group(2))
        baseline = "baseline"
    else:
        # model = re.search(r"model_([^\.]+)\.json", args.file).group(1)
        model = re.search(r"model_([-\w.]+)(?:_num_docs_([-\d]+))?.json", args.file).group(1)

        num_docs = None
        baseline = "letta"

    # evaluate data
    for d in tqdm(data):
        answer = d["true_answers"]
        question = d["question"]
        response = d["letta_responses"]
        if not args.baseline:
            # need to parse response for letta
            response = extract_final_letta_response(response)
        else:
            response = response["response"]

        found = False
        for a in answer:
            if a in response:
                found = True

        if not found and not "INSUFFICIENT INFORMATION" in response:
            # inconclusive: pass to llm judge
            print(question)
            print(answer)
            print(response)
            print(args.baseline)
            doc = "QUESTION: " + question + "\n" + "TRUE ANSWER: " + str(answer) + "\n" + response
            judge = "llm"
            judge_result = evaluate_response(doc)
            print("JUDGEMENT", judge_result)
            if judge_result:
                correct += 1
                found = True
        elif found:
            # answer found in text
            correct += 1
            judge = "text"
        else:
            judge = "text"

        results.append({"question": question, "true_answers": answer, "response": response, "correct": found, "judge": judge})

        total += 1

    # Dump aggregated results
    json.dump(
        {"accuracy": correct / total, "total": total, "results": results},
        open(f"results_{model}_{num_docs}_{baseline}.json", "w"),
        indent=4,
    )
    print(correct / total)
