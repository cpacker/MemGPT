---
title: Example - chat with your data
excerpt: Using Letta to chat with your own data
category: 6580d34ee5e4d00068bf2a1d
---

> 📘 Confirm your installation
>
> Before starting this example, make sure that you've [properly installed Letta](quickstart)

In this example, we're going to use Letta to chat with a custom data source. Specifically, we'll try loading in the Letta research paper and ask Letta questions about it.

### Creating an external data source

To feed external data into a Letta chatbot, we first need to create a data source.

To download the Letta research paper we'll use `curl` (you can also just download the PDF from your browser):

```sh
# we're saving the file as "letta_research_paper.pdf"
curl -L -o letta_research_paper.pdf https://arxiv.org/pdf/2310.08560.pdf
```

Now that we have the paper downloaded, we can create a Letta data source using `letta load`:

```sh
letta load directory --name letta_research_paper --input-files=letta_research_paper.pdf
```

```text
loading data
done loading data
LLM is explicitly disabled. Using MockLLM.
LLM is explicitly disabled. Using MockLLM.
Parsing documents into nodes: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 15/15 [00:00<00:00, 392.09it/s]
Generating embeddings: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 65/65 [00:01<00:00, 37.34it/s]
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 65/65 [00:00<00:00, 388361.48it/s]
Saved local /home/user/.letta/archival/letta_research_paper/nodes.pkl
```

Note: you can ignore the "_LLM is explicitly disabled_" message.

### Attaching the data source to a Letta agent

Now that we've created this data source, we can attach it to a Letta chatbot at any time.

For the sake of this example, let's create a new chatbot using the `letta_doc` persona (but you can use any persona you want):

```sh
# reminder: `letta run --persona letta_doc` will create a new Letta agent using the `letta_doc` persona
letta run --persona letta_doc
```

Once we're chatting with the agent, we can "attach" the data source to the agent's archival memory:

```text
Creating new agent...
Created new agent agent_2.
Hit enter to begin (will request first Letta message)

💭 Looks like a new user just logged in. I need to greet Bob, and provide some context about myself. Let's make this as personable as possible.
🤖 Hello Bob, lovely to meet you! I'm here to assist you with document analysis. How can I be of service today?
> Enter your message: /attach
? Select data source letta_research_paper
Generating embeddings: 0it [00:00, ?it/s]
new size 130
Saved local /home/user/.letta/agents/agent_2/persistence_manager/index/nodes.pkl
Attached data source letta_research_paper to agent agent_2, consisting of 130. Agent now has 130 embeddings in archival memory.
```

### Testing out our new chatbot

Now that the data has been loaded into the chatbot's memory, we can start to ask questions about it:

```text
> Enter your message: The paper I loaded into your archival memory is called Letta. Can you tell me more about it?
💭 The search results show that the Letta paper explores operating-system-inspired techniques to enable large language models (LLMs) to manage memory and achieve unbounded context. The paper evaluates Letta in domains where LLMs are typically limited by finite context windows. This includes document analysis, allowing these models to process lengthy texts beyond their context limits, and conversational agents, where Letta allows for maintaining long-term memory, consistency, and evolution over extended dialogues. Time to relay this to Bob.
🤖 The Letta paper explores novel techniques, inspired by operating systems, to equip large language models (LLMs) with the ability to manage their own memory and thereby handle unbounded context. This system is evaluated in key areas where LLMs are traditionally constrained by limited context: document analysis and conversational agents. In document analysis, Letta effectively handles long texts, surpassing the context limitations of current models. Regarding conversational agents, Letta facilitates maintaining long-term memory, consistency, and adaptability across extensive dialogues.
```

### Loading other data types

In this example, we loaded a single PDF into a chatbots external memory. However Letta supports various types of data, such as full directories of files and even databases - [see the full data sources list](data_sources).
