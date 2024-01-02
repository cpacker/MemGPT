---
title: Example - chat with your data
excerpt: Using MemGPT to chat with your own data
category: 6580d34ee5e4d00068bf2a1d
---

> 📘 Confirm your installation
>
> Before starting this example, make sure that you've [properly installed MemGPT](quickstart)

In this example, we're going to use MemGPT to chat with a custom data source. Specifically, we'll try loading in the MemGPT research paper and ask MemGPT questions about it.

### Creating an external data source

To feed external data into a MemGPT chatbot, we first need to create a data source.

To download the MemGPT research paper we'll use `curl` (you can also just download the PDF from your browser):

```sh
# we're saving the file as "memgpt_research_paper.pdf"
curl -L -o memgpt_research_paper.pdf https://arxiv.org/pdf/2310.08560.pdf
```

Now that we have the paper downloaded, we can create a MemGPT data source using `memgpt load`:

```sh
memgpt load directory --name memgpt_research_paper --input-files=memgpt_research_paper.pdf
```

```text
loading data
done loading data
LLM is explicitly disabled. Using MockLLM.
LLM is explicitly disabled. Using MockLLM.
Parsing documents into nodes: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 15/15 [00:00<00:00, 392.09it/s]
Generating embeddings: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 65/65 [00:01<00:00, 37.34it/s]
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 65/65 [00:00<00:00, 388361.48it/s]
Saved local /home/user/.memgpt/archival/memgpt_research_paper/nodes.pkl
```

Note: you can ignore the "_LLM is explicitly disabled_" message.

### Attaching the data source to a MemGPT agent

Now that we've created this data source, we can attach it to a MemGPT chatbot at any time.

For the sake of this example, let's create a new chatbot using the `memgpt_doc` persona (but you can use any persona you want):

```sh
# reminder: `memgpt run --persona memgpt_doc` will create a new MemGPT agent using the `memgpt_doc` persona
memgpt run --persona memgpt_doc
```

Once we're chatting with the agent, we can "attach" the data source to the agent's archival memory:

```text
Creating new agent...
Created new agent agent_2.
Hit enter to begin (will request first MemGPT message)

💭 Looks like a new user just logged in. I need to greet Bob, and provide some context about myself. Let's make this as personable as possible.
🤖 Hello Bob, lovely to meet you! I'm here to assist you with document analysis. How can I be of service today?
> Enter your message: /attach
? Select data source memgpt_research_paper
Generating embeddings: 0it [00:00, ?it/s]
new size 130
Saved local /home/user/.memgpt/agents/agent_2/persistence_manager/index/nodes.pkl
Attached data source memgpt_research_paper to agent agent_2, consisting of 130. Agent now has 130 embeddings in archival memory.
```

### Testing out our new chatbot

Now that the data has been loaded into the chatbot's memory, we can start to ask questions about it:

```text
> Enter your message: The paper I loaded into your archival memory is called MemGPT. Can you tell me more about it?
💭 The search results show that the MemGPT paper explores operating-system-inspired techniques to enable large language models (LLMs) to manage memory and achieve unbounded context. The paper evaluates MemGPT in domains where LLMs are typically limited by finite context windows. This includes document analysis, allowing these models to process lengthy texts beyond their context limits, and conversational agents, where MemGPT allows for maintaining long-term memory, consistency, and evolution over extended dialogues. Time to relay this to Bob.
🤖 The MemGPT paper explores novel techniques, inspired by operating systems, to equip large language models (LLMs) with the ability to manage their own memory and thereby handle unbounded context. This system is evaluated in key areas where LLMs are traditionally constrained by limited context: document analysis and conversational agents. In document analysis, MemGPT effectively handles long texts, surpassing the context limitations of current models. Regarding conversational agents, MemGPT facilitates maintaining long-term memory, consistency, and adaptability across extensive dialogues.
```

### Loading other data types

In this example, we loaded a single PDF into a chatbots external memory. However MemGPT supports various types of data, such as full directories of files and even databases - [see the full data sources list](data_sources).
