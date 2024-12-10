We evaluate on multi-hop memory edits, to see how well agents are able to assess and resolve inconsistencies in their memory.


Get the data from the [MQuAKE: Assessing Knowledge Editing in Language Models via Multi-Hop Questions](https://arxiv.org/abs/2305.14795)
```
git clone https://github.com/princeton-nlp/MQuAKE.git
```

We use the latest recommended dataset version, which is `MQuAKE/datasets/MQuAKE-CF-3k-v2.json`. 

Then, convert this dataset into our data format, for debugging purposes, also pass in `--num-questions` to run on a smaller set of instances.
```
python convert_mquake_to_explicit_format.py \
    --input_file MQuAKE/datasets/MQuAKE-CF-3k-v2.json \
    --output_file MQuAKE/datasets/letta-MQuAKE-CF-3k-v2-100.json \
```


The output of the previous generated step can then be fed to agents in the Letta framework.

To generate predictions with just the baseline Letta agent, run:

```
python run_fact_editing.py \
    --input_file_name MQuAKE/datasets/letta-MQuAKE-CF-3k-v2-100.json \
    --predictions_file_name MQuAKE/datasets/predictions-letta-MQuAKE-CF-3k-v2-100-fact-prompt-v2-offline-memory.json
```

To run the async memory agent:
```
python run_fact_editing.py \
    --input_file_name MQuAKE/datasets/letta-MQuAKE-CF-3k-v2-100.json \
    --predictions_file_name MQuAKE/datasets/predictions-letta-MQuAKE-CF-3k-v2-100-fact-prompt-v2-offline-memory.json  \
    --offline_memory
```

To evaluate the predictions:
```
python evaluate_fact_edit.py \
    --input_file_name MQuAKE/datasets/letta-MQuAKE-CF-3k-v2-100.json \
    --predictions_file_name MQuAKE/datasets/predictions-letta-MQuAKE-CF-3k-v2-100-fact-prompt.json
```