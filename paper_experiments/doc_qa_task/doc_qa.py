"""
To evaluate Letta's ability to analyze documents, we benchmark Letta against fixed-context
baselines on the retriever-reader document QA task from Liu et al. (2023a). In this task, a question
is selected from the NaturalQuestions-Open dataset, and a retriever selects relevant Wikipedia documents for the question.
A reader model (the LLM) is then fed these documents as input, and is
asked to use the provided documents to answer the question. Similar to Liu et al. (2023a),
we evaluate reader accuracy as the number of retrieved documents K increases. In our evaluation setup, both
the fixed-context baselines and Letta use the same retriever, which selects the top K documents
according using Faiss efficient similarity search (Johnson et al., 2019) (which corresponds to
approximate nearest neighbor search) on OpenAI's text-embedding-3-small embeddings. In
Letta, the entire document set is loaded into archival storage, and the retriever naturally emerges
via the archival storage search functionality (which performs embedding-based similarity search).
In the fixed-context baselines, the top-K documents are fetched using the retriever independently
from the LLM inference, similar to the original retriever-reader setup. We use a dump of Wikipedia
from late 2018, following past work on NaturalQuestions-Open (Izacard & Grave, 2020; Izacard
et al., 2021) We randomly sample a subset of 50 questions for each point in the graph.
"""

import argparse
import json
import os
import uuid
from typing import List

from icml_experiments.utils import get_experiment_config, load_gzipped_file
from openai import OpenAI
from tqdm import tqdm

from letta import utils
from letta.agent_store.storage import StorageConnector, TableType
from letta.cli.cli_config import delete
from letta.config import LettaConfig
from letta.credentials import LettaCredentials
from letta.embeddings import embedding_model
from letta.utils import count_tokens

DATA_SOURCE_NAME = "wikipedia"
DOC_QA_PERSONA = "You are Letta DOC-QA bot. Your job is to answer questions about documents that are stored in your archival memory. The answer to the users question will ALWAYS be in your archival memory, so remember to keep searching if you can't find the answer. Answer the questions as if though the year is 2018."  # TODO decide on a good persona/human
DOC_QA_HUMAN = "The user will ask you questions about documents. Answer them to the best of your ability."

BASELINE_PROMPT = (
    "Answer the question provided according to the list of documents below (some of which might be irrelevant. "
    + "In your response, provide both the answer and the document text from which you determined the answer. "
    + "Format your response with the format 'ANSWER: <YOUR ANSWER>, DOCUMENT: <DOCUMENT TEXT>'. "
    + "If none of the documents provided have the answer to the question, reply with 'INSUFFICIENT INFORMATION'. "
    + "Do NOT provide an answer if you cannot find it in the provided documents. "
    + "Your response will only be considered correct if you provide both the answer and relevant document text, or say 'INSUFFICIENT INFORMATION'."
    + "Answer the question as if though the current year is 2018."
)


MEMGPT_PROMPT = (
    "Search your archival memory to answer the provided question. "
    + "Provide both the answer and the archival memory result from which you determined your answer. "
    + "Format your response with the format 'ANSWER: <YOUR ANSWER>, DOCUMENT: <ARCHIVAL MEMORY TEXT>. "
    + "Your task is to answer the question: "
)


def generate_docqa_baseline_response(
    model: str,  # eg 'gpt-4-0613'
    data_souce_name: str,  # data source containing all relevant documents to put in archival memory
    question: str,  # the question to ask the agent about the data source
    num_documents: int,  # how many documents to put in the prompt
    config: LettaConfig,  # the config to use for the archival memory
) -> List[dict]:
    """Format is from the LITM paper:

    Write a high-quality answer for the given question
    using only the provided search results (some of
    which might be irrelevant).

    Document [1](Title: Asian Americans in science and
    technology) ...
    Document [2](Title: List of Nobel laureates in
    Physics) ...
    Document [3](Title: Scientist) ...
    Document [4](Title: Norwegian Americans) ...
    Document [5](Title: Maria Goeppert Mayer) ...

    Question: who got the first nobel prize in physics
    Answer:
    """

    user_id = uuid.UUID(config.anon_clientid)

    # TODO grab the top N documents using data_source_name
    archival_memory = StorageConnector.get_storage_connector(TableType.PASSAGES, config, user_id)
    archival_memory.disable_write = True  # prevent archival memory writes
    archival_memory.filters = {"data_source": data_souce_name}
    archival_memory.size()
    print(f"Attaching archival memory with {archival_memory.size()} passages")

    # grab the top N documents
    embed_model = embedding_model(config.default_embedding_config)
    embedding = embed_model.get_text_embedding(question)
    passages = archival_memory.query(query=question, query_vec=embedding, top_k=num_documents)
    documents_search_results_sorted_by_relevance = [passage.text for passage in passages]

    # print(f"Top {num_documents} documents: {documents_search_results_sorted_by_relevance}")

    # compute truncation length
    extra_text = BASELINE_PROMPT + f"Question: {question}" + f"Answer:"
    padding = count_tokens(extra_text) + 1000
    truncation_length = int((config.default_llm_config.context_window - padding) / num_documents)
    print("Token size", config.default_llm_config.context_window)
    print(f"Truncation length: {truncation_length}, with padding: {padding}")

    # create the block of text holding all the documents
    documents_block_str = ""
    docs = []
    for i, doc in enumerate(documents_search_results_sorted_by_relevance):
        # only include N documents
        if i >= num_documents:
            break

        doc_prompt = f"Document [{i+1}]: {doc} \n"

        # truncate (that's why the performance goes down as x-axis increases)
        if truncation_length is not None:
            doc_prompt = doc_prompt[:truncation_length]
        docs.append(doc_prompt)

        # add to the block of prompt
        documents_block_str += doc_prompt

    credentials = LettaCredentials().load()
    assert credentials.openai_key is not None, credentials.openai_key

    client = OpenAI(api_key=credentials.openai_key)

    # TODO: determine trunction length, and truncate documents
    content = "\n".join(
        [
            BASELINE_PROMPT,
            "\n",
            documents_block_str,
            "\n",
            f"Question: {question}",
        ]
    )
    total_tokens = count_tokens(content)
    print("Total tokens:", total_tokens, num_documents)
    print(len(documents_search_results_sorted_by_relevance))
    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "user", "content": content},
        ],
        model=model,
    )

    response = chat_completion.choices[0].message.content
    return {"response": response, "documents": docs}
    # return response


def generate_docqa_response(
    config: LettaConfig,
    letta_client: Letta,
    persona: str,
    human: str,
    data_souce_name: str,  # data source containing all relevant documents to put in archival memory
    question: str,  # the question to ask the agent about the data source
) -> List[dict]:
    """Generate a Letta QA response given an input scenario

    Scenario contains:
    - state of the human profile
    - state of the agent profile
    - data source to load into archival memory (that will have the answer to the question)
    """

    utils.DEBUG = True

    # delete agent if exists
    user_id = uuid.UUID(config.anon_clientid)
    agent_name = f"doc_qa_agent_{config.default_llm_config.model}"
    try:
        delete("agent", agent_name)
    except Exception as e:
        print(e)

    # Create a new Agent that models the scenario setup
    agent_state = letta_client.create_agent(
        {
            "name": agent_name,
            "persona": persona,
            "human": human,
            "llm_config": config.default_llm_config,
            "embedding_config": config.default_embedding_config,
        }
    )

    ## Attach the archival memory to the agent
    # attach(agent_state.name, data_source=data_souce_name)
    # HACK: avoid copying all the data by overriding agent archival storage
    archival_memory = StorageConnector.get_storage_connector(TableType.PASSAGES, config, user_id)
    archival_memory.disable_write = True  # prevent archival memory writes
    archival_memory.filters = {"data_source": data_souce_name}
    archival_memory.size()
    print(f"Attaching archival memory with {archival_memory.size()} passages")

    # override the agent's archival memory with table containing wikipedia embeddings
    letta_client.server._get_or_load_agent(user_id, agent_state.id).persistence_manager.archival_memory.storage = archival_memory
    print("Loaded agent")

    ## sanity check: before experiment (agent should have source passages)
    # memory = letta_client.get_agent_memory(agent_state.id)
    # assert memory["archival_memory"] == archival_memory_size, f"Archival memory size is wrong: {memory['archival_memory']}"

    # Run agent.step() / or client.user_message to generate a response from the Letta agent
    prompt_message = " ".join(
        [
            MEMGPT_PROMPT,
            f"{question}?",
        ]
    )
    response = letta_client.user_message(agent_id=agent_state.id, message=prompt_message)

    ## sanity check: after experiment (should NOT have inserted anything into archival)
    # memory = letta_client.get_agent_memory(agent_state.id)
    # assert memory["archival_memory"] == archival_memory_size, f"Archival memory size is wrong: {memory['archival_memory']}"

    # Return that response (may include multiple messages if the agent does retrieval)
    return response


def evaluate_letta_response(letta_responses: List[dict], gold_answers: List[str]) -> bool:
    """Score a Letta response (which is a list of Letta messages) against a gold answer

    We evaluate with the following metric: accuracy
    TODO score with LLM judge?

    NOTE: gold_answers should be length 1, even though it's a list
    """
    raise NotImplementedError


def run_docqa_task(
    model="gpt-4", provider="openai", baseline="letta", num_docs=1, n_samples=50
) -> List[dict]:  # how many samples (questions) from the file
    """Run the full set of Letta doc QA experiments"""

    # Grab the question data
    data_file = "icml_experiments/qa_data/30_total_documents/nq-open-30_total_documents_gold_at_0.jsonl.gz"
    all_question_data = load_gzipped_file(data_file)

    config = get_experiment_config(os.environ.get("PGVECTOR_TEST_DB_URL"), endpoint_type=provider, model=model)
    config.save()  # save config to file

    # result filename
    if baseline == "letta":
        filename = f"results/doc_qa_results_model_{model}.json"
    else:
        filename = f"results/doc_qa_baseline_model_{model}_num_docs_{num_docs}.json"
    print("Results file:", filename)

    if os.path.exists(filename):
        all_response_data = json.load(open(filename, "r"))
    else:
        all_response_data = []

    # letta_client = Letta(config=config)
    letta_client = Letta()
    # letta_client = Letta(quickstart="openai")

    # Loop through and run the doc QA
    count = 0
    cutoff = 50
    for data in tqdm(list(all_question_data)[len(all_response_data) : cutoff]):
        if count > n_samples:
            break

        # Each line in the jsonl.gz has:
        # - a question (str)
        # - a set of answers (List[str]), often len 1
        # - a set of context documents one of which contains the answer (List[dict])
        # - a gold annotation that has a title of the context doc, a long answer, and a list of short answers
        question = data["question"]
        data["ctxs"]
        answers = data["answers"]

        # The only thing we actually use here is the 'question'
        # We ignore the documents, and instead rely on a set of documents that is already in a data source
        # TODO make sure this is correct
        if baseline == "letta":
            responses = generate_docqa_response(
                config=config,
                letta_client=letta_client,
                persona=DOC_QA_PERSONA,
                human=DOC_QA_HUMAN,
                data_souce_name=DATA_SOURCE_NAME,
                question=question,
            )
            prompt = None
        else:
            responses = generate_docqa_baseline_response(
                model=model, data_souce_name=DATA_SOURCE_NAME, question=question, num_documents=num_docs, config=config
            )
            prompt = BASELINE_PROMPT
        # print(responses)

        all_response_data.append(
            {
                "question": question,
                "true_answers": answers,
                "letta_responses": responses,
                "prompt": prompt,
                # "correct": evaluate_letta_response(responses, answers),
            }
        )
        # write to JSON file
        with open(filename, "w") as f:
            json.dump(all_response_data, f, indent=4)

    return all_response_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test script")
    parser.add_argument("--model", type=str, help="The model to use")
    parser.add_argument("--provider", default="openai", type=str, help="The provider to use")
    parser.add_argument("--baseline", default="letta", type=str, help="The baseline to use")
    parser.add_argument("--num_docs", default=5, type=int, help="The number of documents to use in the prompt (baseline-only)")
    args = parser.parse_args()

    results = run_docqa_task(args.model, args.provider, args.baseline, args.num_docs)
