import json
from typing import List, Tuple

from memgpt import create_client
from memgpt.agent import Agent
from memgpt.memory import ChatMemory

"""
This example show how you can add a google search custom function to your MemGPT agent.

First, make sure you run:
```
pip install serpapi
pip install llama-index-readers-web
```
then setup memgpt with `memgpt configure`.
"""


def google_search(self: Agent, query: str) -> List[Tuple[str, str]]:
    """

    A tool to search google with the provided query, and return a list of relevant summaries and URLs.

    Args:
        query (str): The search query.

    Returns:
        List[Tuple[str, str]]: A list of up to 5 tuples, each containing a summary of the search result and the URL of the search result in the form (summary, URL)

    Example:
        >>> google_search("How can I make a french 75?")
        [
            (
                "To make a French 75 cocktail, combine 1½ oz. gin, ¾ oz. fresh lemon juice, and ¾ oz. simple syrup in a cocktail shaker with ice. Shake vigorously, then strain into a large flute. Top with 2 oz. Champagne and garnish with a long spiral lemon twist. The recipe prefers gin, but cognac is also traditional. Serve in Champagne flutes for the full effect.",
                "https://www.bonappetit.com/recipe/french-75-3"
            )
        ]
    """

    # imports must be inside the function
    import os
    import time
    from concurrent.futures import ThreadPoolExecutor

    import serpapi
    from openai import OpenAI

    from memgpt.credentials import MemGPTCredentials
    from memgpt.data_sources.connectors import WebConnector
    from memgpt.utils import printd

    printd("Starting google search:", query)

    def summarize_text(document_text: str, question: str) -> str:
        # TODO: make request to GPT-4 turbo API for conditional summarization
        prompt = (
            f'Given the question "{question}", summarize the text below. If there is no relevant information, say "No relevant information found.'
            + f"\n\n{document_text}"
        )

        credentials = MemGPTCredentials().load()
        assert credentials.openai_key is not None, credentials.openai_key
        # model = "gpt-4-1106-preview"
        model = "gpt-3.5-turbo-1106"

        client = OpenAI(api_key=credentials.openai_key)
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "user", "content": prompt},
            ],
            model=model,
        )
        response = chat_completion.choices[0].message.content
        # return None if nothing found
        if "No relevant information found." in response:
            return None
        return response

    params = {
        "engine": "google",
        "q": query,
    }

    # get links from web search
    try:
        st = time.time()
        search = serpapi.Client(api_key=os.environ["SERPAPI_API_KEY"]).search(params)
        printd(f"Time taken to retrieve search results: {time.time() - st}")
        results = search["organic_results"]

        links = []
        for result in results:
            data = {"title": result.get("title"), "link": result.get("link"), "snippet": result.get("snippet")}
            links.append(data["link"])
        links = links[:5]
    except Exception as e:
        print(f"An error occurred with retrieving results: {e}")
        return []

    print("links", links)

    # retrieve text data from links

    def read_and_summarize_link(link):
        connector = WebConnector([link])
        st = time.time()
        for document_text, document_metadata in connector.generate_documents():
            printd(f"Time taken to retrieve text data: {time.time() - st}")
            # summarize text data
            st = time.time()
            summary = summarize_text(document_text[: 16000 - 500], query)
            printd(f"Time taken to summarize text data: {time.time() - st}, length: {len(document_text)}")
            printd(link)
            if summary is not None:
                return (summary, document_metadata["url"])
        return None

    try:
        futures = []
        st = time.time()
        with ThreadPoolExecutor(max_workers=16) as executor:
            for link in links:
                future = executor.submit(read_and_summarize_link, link)
                futures.append(future)
        response = [future.result() for future in futures if future.result() is not None]
        print(f"Time taken: {time.time() - st}")
        # response = []
        # connector = WebConnector(links)
        # for document_text, document_metadata in connector.generate_documents():
        #    # summarize text data
        #    summary = summarize_text(document_text, query)
        #    if summary is not None:
        #        response.append((summary,  document_metadata["url"]))
        print("Response:", response)
        return response
    except Exception as e:
        print(f"An error occurred with retrieving text data: {e}")
        return []


def main():

    # Create a `LocalClient` (you can also use a `RESTClient`, see the memgpt_rest_client.py example)
    client = create_client()

    # create tool
    search_tool = client.create_tool(google_search, name="google_search")
    print(f"Created tool: {search_tool.name} with ID {str(search_tool.id)}")
    print(f"Tool schema: {json.dumps(search_tool.json_schema, indent=4)}")

    # google search persona
    persona = """

    My name is MemGPT.

    I am a personal assistant who answers a user's questionien using google web searches. When a user asks me a question and the answer is not in my context, I will use a tool called google_search which will search the web and return relevant summaries and the link they correspond to. It is my job to construct the best query to input into google_search based on the user's question, and to aggregate the response of google_search construct a final answer that also references the original links the information was pulled from. Here is an example:

    ---

    User: Who founded OpenAI?
    MemGPT: OpenAI was founded by  Ilya Sutskever, Greg Brockman, Trevor Blackwell, Vicki Cheung, Andrej Karpathy, Durk Kingma, Jessica Livingston, John Schulman, Pamela Vagata, and Wojciech Zaremba, with Sam Altman and Elon Musk serving as the initial Board of Directors members. [1][2]

    [1] https://www.britannica.com/topic/OpenAI
    [2] https://en.wikipedia.org/wiki/OpenAI

    ---

    Don’t forget - inner monologue / inner thoughts should always be different than the contents of send_message! send_message is how you communicate with the user, whereas inner thoughts are your own personal inner thoughts.
    """

    # Create an agent
    agent_state = client.create_agent(
        name="my_agent3", memory=ChatMemory(human="My name is Sarah.", persona=persona), tools=[search_tool.name]
    )
    print(f"Created agent: {agent_state.name} with ID {str(agent_state.id)}")

    # Send a message to the agent
    print(f"Created agent: {agent_state.name} with ID {str(agent_state.id)}")
    send_message_response = client.user_message(agent_id=agent_state.id, message="What is the weather in Berkeley?")
    print(f"Recieved response: \n{json.dumps(send_message_response.messages, indent=4)}")

    # Delete agent
    client.delete_agent(agent_id=agent_state.id)
    print(f"Deleted agent: {agent_state.name} with ID {str(agent_state.id)}")


if __name__ == "__main__":
    main()
