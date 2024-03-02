# type: ignore

import uuid
import typer
import time
from typing import Annotated

from memgpt import create_client
from memgpt.config import MemGPTConfig

# from memgpt.agent import Agent
from memgpt.errors import LLMJSONParsingError
from memgpt.benchmark.constants import TRIES, PERSONA, HUMAN, PROMPTS

app = typer.Typer()


def send_message(message: str, agent_id, turn: int, fn_type: str, print_msg: bool = False, n_tries: int = TRIES):
    try:
        print_msg = f"\t-> Now running {fn_type}. Progress: {turn}/{n_tries}"
        print(print_msg, end="\r", flush=True)
        response = client.user_message(agent_id=agent_id, message=message, return_token_count=True)

        if turn + 1 == n_tries:
            print("  " * len(print_msg), end="\r", flush=True)

        for r in response:
            if "function_call" in r and fn_type in r["function_call"] and any("assistant_message" in re for re in response):
                return True, r["function_call"]

        return False, "No function called."
    except LLMJSONParsingError as e:
        print(f"Error in parsing MemGPT JSON: {e}")
        return False, "Failed to decode valid MemGPT JSON from LLM output."
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return False, "An unexpected error occurred."


@app.command()
def bench(
    print_messages: Annotated[bool, typer.Option("--messages", help="Print functions calls and messages from the agent.")] = False,
    n_tries: Annotated[int, typer.Option("--n-tries", help="Number of benchmark tries to perform for each function.")] = TRIES,
):
    client = create_client()
    print(f"\nDepending on your hardware, this may take up to 30 minutes. This will also create {n_tries * len(PROMPTS)} new agents.\n")
    config = MemGPTConfig.load()
    print(f"version = {config.memgpt_version}")

    total_score, total_tokens_accumulated, elapsed_time = 0, 0, 0

    for fn_type, message in PROMPTS.items():
        score = 0
        start_time_run = time.time()
        bench_id = uuid.uuid4()

        for i in range(n_tries):
            agent = client.create_agent(name=f"benchmark_{bench_id}_agent_{i}", persona=PERSONA, human=HUMAN)

            agent_id = agent.id
            result, msg = send_message(
                message=message, agent_id=agent_id, turn=i, fn_type=fn_type, print_msg=print_messages, n_tries=n_tries
            )

            if print_messages:
                print(f"\t{msg}")

            if result:
                score += 1

            # TODO: add back once we start tracking usage via the client
            # total_tokens_accumulated += tokens_accumulated

        elapsed_time_run = round(time.time() - start_time_run, 2)
        print(f"Score for {fn_type}: {score}/{n_tries}, took {elapsed_time_run} seconds")

        elapsed_time += elapsed_time_run
        total_score += score

    print(f"\nMEMGPT VERSION: {config.memgpt_version}")
    print(f"CONTEXT WINDOW: {config.default_llm_config.context_window}")
    print(f"MODEL WRAPPER: {config.default_llm_config.model_wrapper}")
    print(f"PRESET: {config.preset}")
    print(f"PERSONA: {config.persona}")
    print(f"HUMAN: {config.human}")

    print(
        # f"\n\t-> Total score: {total_score}/{len(PROMPTS) * n_tries}, took {elapsed_time} seconds at average of {round(total_tokens_accumulated/elapsed_time, 2)} t/s\n"
        f"\n\t-> Total score: {total_score}/{len(PROMPTS) * n_tries}, took {elapsed_time} seconds\n"
    )
