
## Nested K/V (`nested_kv_task`)
This task runs K/V lookups on synthetic data. You can run it with `icml_experiments/nested_kv_task/run.sh`.

## Document Q/A (`doc_qa_task`)
This task runs question answering on a set of embedded wikipedia passages.

### Setup
You need a a running postgres database to run this experiment and an OpenAI account. Set your enviornment variables:
```
export PGVECTOR_TEST_DB_URL=postgresql+pg8000://{username}:{password}@localhost:8888/{db}
export OPENAI_API_KEY={key}
```

## Download data
Download the wikipedia embedding at:
```
huggingface-cli download nlpkevinl/wikipedia_openai_embeddings --repo-type dataset
```

## Loading embeddings
Run the script `./0_load_embeddings.sh`.

This step will take a while. You can check the status of the loading by connecting to `psql`:
```
> psql -h localhost -p {password} -U {username} -d {db}
> SELECT COUNT(*) from letta_passages;
```
Once completed, there will be ~19 million rows in the database.

### Creating an index
To avoid extremeley slow queries, you need to create an index:
```
CREATE INDEX ON letta_passages USING hnsw (embedding vector_l2_ops);
```
You can check to see if the index was created successfully with:
```
> SELECT indexname, indexdef FROM pg_indexes WHERE tablename = 'letta_passages';

letta_passages_embedding_idx | CREATE INDEX letta_passages_embedding_idx ON public.letta_passages USING hnsw (embedding vector_cosine_ops) WITH (m='24', ef_construction='100')
```

## Running Document Q/A
Run the script `./1_run_docqa.sh {model_name} {n_docs} {letta/model_name}`.

## Evaluation
Run the script `./2_run_eval.sh`.
