#!/bin/bash

source "batch_jobs/_experiment_configuration.sh"

mkdir -p ${persistent_dir}/logs/data
mkdir -p ${persistent_dir}/logs/err


all_models=(
    'BertModel'
    'LlamaModel'
    'MistralModel'
)

datasets=(
    "mathematics"
    "english"
    "superuser"
    "stackoverflow"
)


experiment="STE"




for dataset in ${datasets[@]}; do
    for model in ${all_models[@]}; do
        IFS=' ' read -r -a model_names <<< "${model_to_model_name_or_path[$model]}"
        for model_name_or_path in "${model_names[@]}"; do
            short_model_name_or_path=${model_name_or_path_to_short[${model_name_or_path}]}
            experiment_id="encode-textual-Z_${experiment}-${dataset}_m-${short_model_name_or_path}_encode"
            echo ${experiment_id}
                sbatch \
                    -J ${experiment_id} \
                    -o ${persistent_dir}/logs/data/%x.%j.out \
                    -e ${persistent_dir}/logs/err/%x.%j.err \
                    isambard.sh src/data/STE_encode_threshold_memory_safe_userid.py \
                        --input_dir ${persistent_dir}/data/${experiment}/raw/${dataset}/ \
                        --output_dir ${persistent_dir}/data/${experiment}/${dataset}/${model_name_or_path}/ \
                        --model ${model} \
                        --model_name_or_path ${model_name_or_path} \
                        --shuffle False \
                        --dataset_name ${dataset}
        done
    done
done


