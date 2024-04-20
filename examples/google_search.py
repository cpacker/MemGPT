import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple

import serpapi
from openai import OpenAI

from memgpt.credentials import MemGPTCredentials
from memgpt.data_sources.connectors import WebConnector
from memgpt.utils import printd

"""
This example show how you can add a google search custom function to your MemGPT agent.

First, make sure you run `pip install serpapi`, then setup memgpt:

1. Copy this file into the `~/.memgpt/functions` directory:
```
cp examples/google_search.py  ~/.memgpt/functions/google_search.py
```

2. Create a preset file that include the function `google_search`

3. Add the preset file via the CLI:
```
memgpt add preset -f  examples/google_search_preset.yaml --name search_preset
```

4. Run memgpt with the `google_search_persona` persona:
```
memgpt run --preset search_preset --persona google_search_persona
```
"""


def google_search(self, query: str) -> List[Tuple[str, str]]:
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
        printd(f"An error occurred with retrieving results: {e}")
        return []

    # retrieve text data from links

    def read_and_summarize_link(link):
        connector = WebConnector([link])
        st = time.time()
        for document_text, document_metadata in connector.generate_documents():
            printd(f"Time taken to retrieve text data: {time.time() - st}")
            # summarize text data
            st = time.time()
            summary = summarize_text(document_text, query)
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
        printd(f"Time taken: {time.time() - st}")
        # response = []
        # connector = WebConnector(links)
        # for document_text, document_metadata in connector.generate_documents():
        #    # summarize text data
        #    summary = summarize_text(document_text, query)
        #    if summary is not None:
        #        response.append((summary,  document_metadata["url"]))
        return response
    except Exception as e:
        printd(f"An error occurred with retrieving text data: {e}")
        return []
