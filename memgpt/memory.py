from abc import ABC, abstractmethod
import os
import datetime
import re
from typing import Optional, List, Tuple

from memgpt.constants import MESSAGE_SUMMARY_WARNING_FRAC, MEMGPT_DIR
from memgpt.utils import cosine_similarity, get_local_time, printd, count_tokens
from memgpt.prompts.gpt_summarize import SYSTEM as SUMMARY_PROMPT_SYSTEM
from memgpt import utils
from memgpt.openai_tools import get_embedding_with_backoff, chat_completion_with_backoff
from llama_index import (
    VectorStoreIndex,
    EmptyIndex,
    get_response_synthesizer,
    load_index_from_storage,
    StorageContext,
    Document,
)
from llama_index.node_parser import SimpleNodeParser
from llama_index.node_parser import SimpleNodeParser
from llama_index.retrievers import VectorIndexRetriever
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.indices.postprocessor import SimilarityPostprocessor

from memgpt.embeddings import embedding_model
from memgpt.config import MemGPTConfig

from memgpt.embeddings import embedding_model
from memgpt.config import MemGPTConfig


class CoreMemory(object):
    """Held in-context inside the system message

    Core Memory: Refers to the system block, which provides essential, foundational context to the AI.
    This includes the persona information, essential user details,
    and any other baseline data you deem necessary for the AI's basic functioning.
    """

    def __init__(self, persona=None, human=None, persona_char_limit=None, human_char_limit=None, archival_memory_exists=True):
        self.persona = persona
        self.human = human
        self.persona_char_limit = persona_char_limit
        self.human_char_limit = human_char_limit

        # affects the error message the AI will see on overflow inserts
        self.archival_memory_exists = archival_memory_exists

    def __repr__(self) -> str:
        return f"\n### CORE MEMORY ###" + f"\n=== Persona ===\n{self.persona}" + f"\n\n=== Human ===\n{self.human}"

    def to_dict(self):
        return {
            "persona": self.persona,
            "human": self.human,
        }

    @classmethod
    def load(cls, state):
        return cls(state["persona"], state["human"])

    def edit_persona(self, new_persona):
        if self.persona_char_limit and len(new_persona) > self.persona_char_limit:
            error_msg = f"Edit failed: Exceeds {self.persona_char_limit} character limit (requested {len(new_persona)})."
            if self.archival_memory_exists:
                error_msg = f"{error_msg} Consider summarizing existing core memories in 'persona' and/or moving lower priority content to archival memory to free up space in core memory, then trying again."
            raise ValueError(error_msg)

        self.persona = new_persona
        return len(self.persona)

    def edit_human(self, new_human):
        if self.human_char_limit and len(new_human) > self.human_char_limit:
            error_msg = f"Edit failed: Exceeds {self.human_char_limit} character limit (requested {len(new_human)})."
            if self.archival_memory_exists:
                error_msg = f"{error_msg} Consider summarizing existing core memories in 'human' and/or moving lower priority content to archival memory to free up space in core memory, then trying again."
            raise ValueError(error_msg)

        self.human = new_human
        return len(self.human)

    def edit(self, field, content):
        if field == "persona":
            return self.edit_persona(content)
        elif field == "human":
            return self.edit_human(content)
        else:
            raise KeyError

    def edit_append(self, field, content, sep="\n"):
        if field == "persona":
            new_content = self.persona + sep + content
            return self.edit_persona(new_content)
        elif field == "human":
            new_content = self.human + sep + content
            return self.edit_human(new_content)
        else:
            raise KeyError

    def edit_replace(self, field, old_content, new_content):
        if field == "persona":
            if old_content in self.persona:
                new_persona = self.persona.replace(old_content, new_content)
                return self.edit_persona(new_persona)
            else:
                raise ValueError("Content not found in persona (make sure to use exact string)")
        elif field == "human":
            if old_content in self.human:
                new_human = self.human.replace(old_content, new_content)
                return self.edit_human(new_human)
            else:
                raise ValueError("Content not found in human (make sure to use exact string)")
        else:
            raise KeyError


def summarize_messages(
    agent_config,
    message_sequence_to_summarize,
):
    """Summarize a message sequence using GPT"""
    # we need the context_window
    context_window = agent_config.context_window

    summary_prompt = SUMMARY_PROMPT_SYSTEM
    summary_input = str(message_sequence_to_summarize)
    summary_input_tkns = count_tokens(summary_input)
    if summary_input_tkns > MESSAGE_SUMMARY_WARNING_FRAC * context_window:
        trunc_ratio = (MESSAGE_SUMMARY_WARNING_FRAC * context_window / summary_input_tkns) * 0.8  # For good measure...
        cutoff = int(len(message_sequence_to_summarize) * trunc_ratio)
        summary_input = str(
            [summarize_messages(agent_config, context_window, message_sequence_to_summarize[:cutoff])]
            + message_sequence_to_summarize[cutoff:]
        )
    message_sequence = [
        {"role": "system", "content": summary_prompt},
        {"role": "user", "content": summary_input},
    ]

    response = chat_completion_with_backoff(
        agent_config=agent_config,
        messages=message_sequence,
    )

    printd(f"summarize_messages gpt reply: {response.choices[0]}")
    reply = response.choices[0].message.content
    return reply


class ArchivalMemory(ABC):
    @abstractmethod
    def insert(self, memory_string):
        """Insert new archival memory

        :param memory_string: Memory string to insert
        :type memory_string: str
        """
        pass

    @abstractmethod
    def search(self, query_string, count=None, start=None) -> Tuple[List[str], int]:
        """Search archival memory

        :param query_string: Query string
        :type query_string: str
        :param count: Number of results to return (None for all)
        :type count: Optional[int]
        :param start: Offset to start returning results from (None if 0)
        :type start: Optional[int]

        :return: Tuple of (list of results, total number of results)
        """
        pass

    @abstractmethod
    def __repr__(self) -> str:
        pass


class DummyArchivalMemory(ArchivalMemory):
    """Dummy in-memory version of an archival memory database (eg run on MongoDB)

    Archival Memory: A more structured and deep storage space for the AI's reflections,
    insights, or any other data that doesn't fit into the active memory but
    is essential enough not to be left only to the recall memory.
    """

    def __init__(self, archival_memory_database=None):
        self._archive = [] if archival_memory_database is None else archival_memory_database  # consists of {'content': str} dicts

    def __len__(self):
        return len(self._archive)

    def __repr__(self) -> str:
        if len(self._archive) == 0:
            memory_str = "<empty>"
        else:
            memory_str = "\n".join([d["content"] for d in self._archive])
        return f"\n### ARCHIVAL MEMORY ###" + f"\n{memory_str}"

    def insert(self, memory_string):
        self._archive.append(
            {
                # can eventually upgrade to adding semantic tags, etc
                "timestamp": get_local_time(),
                "content": memory_string,
            }
        )

    def search(self, query_string, count=None, start=None):
        """Simple text-based search"""
        # in the dummy version, run an (inefficient) case-insensitive match search
        # printd(f"query_string: {query_string}")
        matches = [s for s in self._archive if query_string.lower() in s["content"].lower()]
        # printd(f"archive_memory.search (text-based): search for query '{query_string}' returned the following results (limit 5):\n{[str(d['content']) d in matches[:5]]}")
        printd(
            f"archive_memory.search (text-based): search for query '{query_string}' returned the following results (limit 5):\n{[matches[start:count]]}"
        )

        # start/count support paging through results
        if start is not None and count is not None:
            return matches[start : start + count], len(matches)
        elif start is None and count is not None:
            return matches[:count], len(matches)
        elif start is not None and count is None:
            return matches[start:], len(matches)
        else:
            return matches, len(matches)


class DummyArchivalMemoryWithEmbeddings(DummyArchivalMemory):
    """Same as dummy in-memory archival memory, but with bare-bones embedding support"""

    def __init__(self, archival_memory_database=None, embedding_model="text-embedding-ada-002"):
        self._archive = [] if archival_memory_database is None else archival_memory_database  # consists of {'content': str} dicts
        self.embedding_model = embedding_model

    def __len__(self):
        return len(self._archive)

    def _insert(self, memory_string, embedding):
        # Get the embedding
        embedding_meta = {"model": self.embedding_model}
        printd(f"Got an embedding, type {type(embedding)}, len {len(embedding)}")

        self._archive.append(
            {
                "timestamp": get_local_time(),
                "content": memory_string,
                "embedding": embedding,
                "embedding_metadata": embedding_meta,
            }
        )

    def insert(self, memory_string):
        embedding = get_embedding_with_backoff(memory_string, model=self.embedding_model)
        return self._insert(memory_string, embedding)

    def search(self, query_string, count, start):
        """Simple embedding-based search (inefficient, no caching)"""
        # see: https://github.com/openai/openai-cookbook/blob/main/examples/Semantic_text_search_using_embeddings.ipynb
        query_embedding = get_embedding_with_backoff(query_string, model=self.embedding_model)

        # query_embedding = get_embedding(query_string, model=self.embedding_model)
        # our wrapped version supports backoff/rate-limits
        similarity_scores = [cosine_similarity(memory["embedding"], query_embedding) for memory in self._archive]

        # Sort the archive based on similarity scores
        sorted_archive_with_scores = sorted(
            zip(self._archive, similarity_scores),
            key=lambda pair: pair[1],  # Sort by the similarity score
            reverse=True,  # We want the highest similarity first
        )
        printd(
            f"archive_memory.search (vector-based): search for query '{query_string}' returned the following results (limit 5) and scores:\n{str([str(t[0]['content']) + '- score ' + str(t[1]) for t in sorted_archive_with_scores[:5]])}"
        )

        # Extract the sorted archive without the scores
        matches = [item[0] for item in sorted_archive_with_scores]

        # start/count support paging through results
        if start is not None and count is not None:
            return matches[start : start + count], len(matches)
        elif start is None and count is not None:
            return matches[:count], len(matches)
        elif start is not None and count is None:
            return matches[start:], len(matches)
        else:
            return matches, len(matches)


class DummyArchivalMemoryWithFaiss(DummyArchivalMemory):
    """Dummy in-memory version of an archival memory database, using a FAISS
    index for fast nearest-neighbors embedding search.

    Archival memory is effectively "infinite" overflow for core memory,
    and is read-only via string queries.

    Archival Memory: A more structured and deep storage space for the AI's reflections,
    insights, or any other data that doesn't fit into the active memory but
    is essential enough not to be left only to the recall memory.
    """

    def __init__(self, index=None, archival_memory_database=None, embedding_model="text-embedding-ada-002", k=100):
        if index is None:
            import faiss

            self.index = faiss.IndexFlatL2(1536)  # openai embedding vector size.
        else:
            self.index = index
        self.k = k
        self._archive = [] if archival_memory_database is None else archival_memory_database  # consists of {'content': str} dicts
        self.embedding_model = embedding_model
        self.embeddings_dict = {}
        self.search_results = {}

    def __len__(self):
        return len(self._archive)

    def insert(self, memory_string):
        import numpy as np

        # Get the embedding
        embedding = get_embedding_with_backoff(memory_string, model=self.embedding_model)

        print(f"Got an embedding, type {type(embedding)}, len {len(embedding)}")

        self._archive.append(
            {
                # can eventually upgrade to adding semantic tags, etc
                "timestamp": get_local_time(),
                "content": memory_string,
            }
        )
        embedding = np.array([embedding]).astype("float32")
        self.index.add(embedding)

    def search(self, query_string, count=None, start=None):
        """Simple embedding-based search (inefficient, no caching)"""
        # see: https://github.com/openai/openai-cookbook/blob/main/examples/Semantic_text_search_using_embeddings.ipynb

        # query_embedding = get_embedding(query_string, model=self.embedding_model)
        # our wrapped version supports backoff/rate-limits
        import numpy as np

        if query_string in self.embeddings_dict:
            search_result = self.search_results[query_string]
        else:
            query_embedding = get_embedding_with_backoff(query_string, model=self.embedding_model)
            _, indices = self.index.search(np.array([np.array(query_embedding, dtype=np.float32)]), self.k)
            search_result = [self._archive[idx] if idx < len(self._archive) else "" for idx in indices[0]]
            self.embeddings_dict[query_string] = query_embedding
            self.search_results[query_string] = search_result

        if start is not None and count is not None:
            toprint = search_result[start : start + count]
        else:
            if len(search_result) >= 5:
                toprint = search_result[:5]
            else:
                toprint = search_result
        printd(
            f"archive_memory.search (vector-based): search for query '{query_string}' returned the following results ({start}--{start+5}/{len(search_result)}) and scores:\n{str([t[:60] if len(t) > 60 else t for t in toprint])}"
        )

        # Extract the sorted archive without the scores
        matches = search_result

        # start/count support paging through results
        if start is not None and count is not None:
            return matches[start : start + count], len(matches)
        elif start is None and count is not None:
            return matches[:count], len(matches)
        elif start is not None and count is None:
            return matches[start:], len(matches)
        else:
            return matches, len(matches)


class RecallMemory(ABC):
    @abstractmethod
    def text_search(self, query_string, count=None, start=None):
        pass

    @abstractmethod
    def date_search(self, query_string, count=None, start=None):
        pass

    @abstractmethod
    def __repr__(self) -> str:
        pass


class DummyRecallMemory(RecallMemory):
    """Dummy in-memory version of a recall memory database (eg run on MongoDB)

    Recall memory here is basically just a full conversation history with the user.
    Queryable via string matching, or date matching.

    Recall Memory: The AI's capability to search through past interactions,
    effectively allowing it to 'remember' prior engagements with a user.
    """

    def __init__(self, message_database=None, restrict_search_to_summaries=False):
        self._message_logs = [] if message_database is None else message_database  # consists of full message dicts

        # If true, the pool of messages that can be queried are the automated summaries only
        # (generated when the conversation window needs to be shortened)
        self.restrict_search_to_summaries = restrict_search_to_summaries

    def __len__(self):
        return len(self._message_logs)

    def __repr__(self) -> str:
        # don't dump all the conversations, just statistics
        system_count = user_count = assistant_count = function_count = other_count = 0
        for msg in self._message_logs:
            role = msg["message"]["role"]
            if role == "system":
                system_count += 1
            elif role == "user":
                user_count += 1
            elif role == "assistant":
                assistant_count += 1
            elif role == "function":
                function_count += 1
            else:
                other_count += 1
        memory_str = (
            f"Statistics:"
            + f"\n{len(self._message_logs)} total messages"
            + f"\n{system_count} system"
            + f"\n{user_count} user"
            + f"\n{assistant_count} assistant"
            + f"\n{function_count} function"
            + f"\n{other_count} other"
        )
        return f"\n### RECALL MEMORY ###" + f"\n{memory_str}"

    def insert(self, message):
        raise NotImplementedError("This should be handled by the PersistenceManager, recall memory is just a search layer on top")

    def text_search(self, query_string, count=None, start=None):
        # in the dummy version, run an (inefficient) case-insensitive match search
        message_pool = [d for d in self._message_logs if d["message"]["role"] not in ["system", "function"]]

        printd(
            f"recall_memory.text_search: searching for {query_string} (c={count}, s={start}) in {len(self._message_logs)} total messages"
        )
        matches = [
            d for d in message_pool if d["message"]["content"] is not None and query_string.lower() in d["message"]["content"].lower()
        ]
        printd(f"recall_memory - matches:\n{matches[start:start+count]}")

        # start/count support paging through results
        if start is not None and count is not None:
            return matches[start : start + count], len(matches)
        elif start is None and count is not None:
            return matches[:count], len(matches)
        elif start is not None and count is None:
            return matches[start:], len(matches)
        else:
            return matches, len(matches)

    def _validate_date_format(self, date_str):
        """Validate the given date string in the format 'YYYY-MM-DD'."""
        try:
            datetime.datetime.strptime(date_str, "%Y-%m-%d")
            return True
        except (ValueError, TypeError):
            return False

    def _extract_date_from_timestamp(self, timestamp):
        """Extracts and returns the date from the given timestamp."""
        # Extracts the date (ignoring the time and timezone)
        match = re.match(r"(\d{4}-\d{2}-\d{2})", timestamp)
        return match.group(1) if match else None

    def date_search(self, start_date, end_date, count=None, start=None):
        message_pool = [d for d in self._message_logs if d["message"]["role"] not in ["system", "function"]]

        # First, validate the start_date and end_date format
        if not self._validate_date_format(start_date) or not self._validate_date_format(end_date):
            raise ValueError("Invalid date format. Expected format: YYYY-MM-DD")

        # Convert dates to datetime objects for comparison
        start_date_dt = datetime.datetime.strptime(start_date, "%Y-%m-%d")
        end_date_dt = datetime.datetime.strptime(end_date, "%Y-%m-%d")

        # Next, match items inside self._message_logs
        matches = [
            d
            for d in message_pool
            if start_date_dt <= datetime.datetime.strptime(self._extract_date_from_timestamp(d["timestamp"]), "%Y-%m-%d") <= end_date_dt
        ]

        # start/count support paging through results
        if start is not None and count is not None:
            return matches[start : start + count], len(matches)
        elif start is None and count is not None:
            return matches[:count], len(matches)
        elif start is not None and count is None:
            return matches[start:], len(matches)
        else:
            return matches, len(matches)


class DummyRecallMemoryWithEmbeddings(DummyRecallMemory):
    """Lazily manage embeddings by keeping a string->embed dict"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.embeddings = dict()
        self.embedding_model = "text-embedding-ada-002"
        self.only_use_preloaded_embeddings = False

    def text_search(self, query_string, count, start):
        # in the dummy version, run an (inefficient) case-insensitive match search
        message_pool = [d for d in self._message_logs if d["message"]["role"] not in ["system", "function"]]

        # first, go through and make sure we have all the embeddings we need
        message_pool_filtered = []
        for d in message_pool:
            message_str = d["message"]["content"]
            if self.only_use_preloaded_embeddings:
                if message_str not in self.embeddings:
                    printd(f"recall_memory.text_search -- '{message_str}' was not in embedding dict, skipping.")
                else:
                    message_pool_filtered.append(d)
            elif message_str not in self.embeddings:
                printd(f"recall_memory.text_search -- '{message_str}' was not in embedding dict, computing now")
                self.embeddings[message_str] = get_embedding_with_backoff(message_str, model=self.embedding_model)
                message_pool_filtered.append(d)

        # our wrapped version supports backoff/rate-limits
        query_embedding = get_embedding_with_backoff(query_string, model=self.embedding_model)
        similarity_scores = [cosine_similarity(self.embeddings[d["message"]["content"]], query_embedding) for d in message_pool_filtered]

        # Sort the archive based on similarity scores
        sorted_archive_with_scores = sorted(
            zip(message_pool_filtered, similarity_scores),
            key=lambda pair: pair[1],  # Sort by the similarity score
            reverse=True,  # We want the highest similarity first
        )
        printd(
            f"recall_memory.text_search (vector-based): search for query '{query_string}' returned the following results (limit 5) and scores:\n{str([str(t[0]['message']['content']) + '- score ' + str(t[1]) for t in sorted_archive_with_scores[:5]])}"
        )

        # Extract the sorted archive without the scores
        matches = [item[0] for item in sorted_archive_with_scores]

        # start/count support paging through results
        if start is not None and count is not None:
            return matches[start : start + count], len(matches)
        elif start is None and count is not None:
            return matches[:count], len(matches)
        elif start is not None and count is None:
            return matches[start:], len(matches)
        else:
            return matches, len(matches)


class LocalArchivalMemory(ArchivalMemory):
    """Archival memory built on top of Llama Index"""

    def __init__(self, agent_config, top_k: Optional[int] = 100):
        """Init function for archival memory

        :param archiva_memory_database: name of dataset to pre-fill archival with
        :type archival_memory_database: str
        """

        self.top_k = top_k
        self.agent_config = agent_config

        # locate saved index
        # if self.agent_config.data_source is not None:  # connected data source
        #    directory = f"{MEMGPT_DIR}/archival/{self.agent_config.data_source}"
        #    assert os.path.exists(directory), f"Archival memory database {self.agent_config.data_source} does not exist"
        # elif self.agent_config.name is not None:
        if self.agent_config.name is not None:
            directory = agent_config.save_agent_index_dir()
            if not os.path.exists(directory):
                # no existing archival storage
                directory = None

        # load/create index
        if directory:
            storage_context = StorageContext.from_defaults(persist_dir=directory)
            self.index = load_index_from_storage(storage_context)
        else:
            self.index = EmptyIndex()

        # create retriever
        if isinstance(self.index, EmptyIndex):
            self.retriever = None  # cant create retriever over empty indes
        else:
            self.retriever = VectorIndexRetriever(
                index=self.index,  # does this get refreshed?
                similarity_top_k=self.top_k,
            )

        # TODO: have some mechanism for cleanup otherwise will lead to OOM
        self.cache = {}

    def save(self):
        """Save the index to disk"""
        # if self.agent_config.data_sources:  # update original archival index
        #    # TODO: this corrupts the originally loaded data. do we want to do this?
        #    utils.save_index(self.index, self.agent_config.data_sources)
        # else:

        # don't need to save data source, since we assume data source data is already loaded into the agent index
        utils.save_agent_index(self.index, self.agent_config)

    def insert(self, memory_string):
        self.index.insert(memory_string)

        # TODO: figure out if this needs to be refreshed (probably not)
        self.retriever = VectorIndexRetriever(
            index=self.index,
            similarity_top_k=self.top_k,
        )

    def search(self, query_string, count=None, start=None):
        print("searching with local")
        if self.retriever is None:
            print("Warning: archival memory is empty")
            return [], 0

        start = start if start else 0
        count = count if count else self.top_k
        count = min(count + start, self.top_k)

        if query_string not in self.cache:
            self.cache[query_string] = self.retriever.retrieve(query_string)

        results = self.cache[query_string][start : start + count]
        results = [{"timestamp": get_local_time(), "content": node.node.text} for node in results]
        # from pprint import pprint
        # pprint(results)
        return results, len(results)

    def __repr__(self) -> str:
        if isinstance(self.index, EmptyIndex):
            memory_str = "<empty>"
        else:
            memory_str = self.index.ref_doc_info
        return f"\n### ARCHIVAL MEMORY ###" + f"\n{memory_str}"


class EmbeddingArchivalMemory(ArchivalMemory):
    """Archival memory with embedding based search"""

    def __init__(self, agent_config, top_k: Optional[int] = 100):
        """Init function for archival memory

        :param archival_memory_database: name of dataset to pre-fill archival with
        :type archival_memory_database: str
        """
        from memgpt.connectors.storage import StorageConnector

        self.top_k = top_k
        self.agent_config = agent_config
        config = MemGPTConfig.load()

        # create embedding model
        self.embed_model = embedding_model()
        self.embedding_chunk_size = config.embedding_chunk_size

        # create storage backend
        self.storage = StorageConnector.get_storage_connector(agent_config=agent_config)
        # TODO: have some mechanism for cleanup otherwise will lead to OOM
        self.cache = {}

    def save(self):
        """Save the index to disk"""
        self.storage.save()

    def insert(self, memory_string):
        """Embed and save memory string"""
        from memgpt.connectors.storage import Passage

        try:
            passages = []

            # create parser
            parser = SimpleNodeParser.from_defaults(chunk_size=self.embedding_chunk_size)

            # breakup string into passages
            for node in parser.get_nodes_from_documents([Document(text=memory_string)]):
                embedding = self.embed_model.get_text_embedding(node.text)
                passages.append(Passage(text=node.text, embedding=embedding, doc_id=f"agent_{self.agent_config.name}_memory"))

            # insert passages
            self.storage.insert_many(passages)
            return True
        except Exception as e:
            print("Archival insert error", e)
            raise e

    def search(self, query_string, count=None, start=None):
        """Search query string"""
        try:
            if query_string not in self.cache:
                # self.cache[query_string] = self.retriever.retrieve(query_string)
                query_vec = self.embed_model.get_text_embedding(query_string)
                self.cache[query_string] = self.storage.query(query_string, query_vec, top_k=self.top_k)

            start = start if start else 0
            count = count if count else self.top_k
            end = min(count + start, len(self.cache[query_string]))

            results = self.cache[query_string][start:end]
            results = [{"timestamp": get_local_time(), "content": node.text} for node in results]
            return results, len(results)
        except Exception as e:
            print("Archival search error", e)
            raise e

    def __repr__(self) -> str:
        limit = 10
        passages = []
        for passage in list(self.storage.get_all(limit)):  # TODO: only get first 10
            passages.append(str(passage.text))
        memory_str = "\n".join(passages)
        return f"\n### ARCHIVAL MEMORY ###" + f"\n{memory_str}"

    def __len__(self):
        return self.storage.size()
