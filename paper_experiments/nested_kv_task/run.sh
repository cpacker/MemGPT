for nest in 4 3 2 1
do
for model in "gpt-3.5-turbo-1106" "gpt-4-0613" "gpt-4-1106-preview"
do
    for seed in 0 1 2 3 4 5 6 7 8 9 10
    do
        for baseline in $model "letta"
        do
            python icml_experiments/nested_kv_task/nested_kv.py --model $model  --task kv_nested --baseline $baseline --nesting_levels $nest --seed $seed #--rerun
        done
    done
done
done
