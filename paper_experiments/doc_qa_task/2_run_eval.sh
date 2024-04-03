docs=(1 5 10 20 50 100 200 700)
models=("gpt-4-0613" "gpt-3.5-turbo-1106" "gpt-4-1106-preview")

## run memgpt eval
for model in  "${models[@]}";
do
  poetry run python icml_experiments/doc_qa_task/llm_judge_doc_qa.py --file results/doc_qa_results_model_${model}.json
done

# Iterate over each model
for model in "${models[@]}"; do
    # Iterate over each doc
    for doc in "${docs[@]}"; do
        # Construct and run the command
        echo "Running for model $model with $doc docs..."
        poetry run python icml_experiments/doc_qa_task/llm_judge_doc_qa.py --file results/doc_qa_baseline_model_${model}_num_docs_${doc}.json --baseline
    done
done
