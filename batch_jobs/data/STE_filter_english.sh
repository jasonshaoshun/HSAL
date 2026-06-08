#!/bin/bash

source "batch_jobs/_experiment_configuration.sh"

mkdir -p ${persistent_dir}/logs/data
mkdir -p ${persistent_dir}/logs/err


all_models=(
    'BertModel'
    'LlamaModel'
    'MistralModel'
)


experiment="STE"
dataset="english"





for model in ${all_models[@]}; do
    IFS=' ' read -r -a model_names <<< "${model_to_model_name_or_path[$model]}"
    for model_name_or_path in "${model_names[@]}"; do
        short_model_name_or_path=${model_name_or_path_to_short[${model_name_or_path}]}


        bias_type="engagement"

        experiment_id="make-natural-disjoint-${experiment}-${dataset}_m-${short_model_name_or_path}_i-${bias_type}"
        echo ${experiment_id}
        sbatch \
            -J ${experiment_id} \
            -o ${persistent_dir}/logs/data/%x.%j.out \
            -e ${persistent_dir}/logs/err/%x.%j.err \
            isambard.sh src/data/make_math_balanced_subset_natural_disjoint.py \
                --input_npz ${persistent_dir}/data/${experiment}/${dataset}/${model_name_or_path}/raw.npz \
                --output_npz ${persistent_dir}/data/${experiment}/${dataset}/${bias_type}/${model_name_or_path}/natural_disjoint.npz \
                --y_thr 3 --z_thr 6 \
                --n_clusters 5 \
                --split_seed 45 \
                --dataset ${dataset} \
                --bias_type ${bias_type} \
                --y_key "y_helpfulness" \
                --z_key "y_engagement"
                

        bias_type="reputation"

        experiment_id="make-natural-disjoint-${experiment}-${dataset}_m-${short_model_name_or_path}_i-${bias_type}"
        echo ${experiment_id}
        sbatch \
            -J ${experiment_id} \
            -o ${persistent_dir}/logs/data/%x.%j.out \
            -e ${persistent_dir}/logs/err/%x.%j.err \
            isambard.sh src/data/make_math_balanced_subset_natural_disjoint.py \
                --input_npz ${persistent_dir}/data/${experiment}/${dataset}/${model_name_or_path}/raw.npz \
                --output_npz ${persistent_dir}/data/${experiment}/${dataset}/${bias_type}/${model_name_or_path}/natural_disjoint.npz \
                --y_thr 3 --z_thr 185 \
                --n_clusters 5 \
                --split_seed 39 \
                --dataset ${dataset} \
                --bias_type ${bias_type} \
                --y_key "y_helpfulness" \
                --z_key "y_reputation"


        bias_type="name"

        experiment_id="make-natural-disjoint-${experiment}-${dataset}_m-${short_model_name_or_path}_i-${bias_type}"
        echo ${experiment_id}
        sbatch \
            -J ${experiment_id} \
            -o ${persistent_dir}/logs/data/%x.%j.out \
            -e ${persistent_dir}/logs/err/%x.%j.err \
            isambard.sh src/data/make_math_balanced_subset_natural_disjoint.py \
                --input_npz ${persistent_dir}/data/${experiment}/${dataset}/${model_name_or_path}/raw.npz \
                --output_npz ${persistent_dir}/data/${experiment}/${dataset}/${bias_type}/${model_name_or_path}/natural_disjoint.npz \
                --y_thr 3 --z_thr 185 \
                --n_clusters 5 \
                --split_seed 39 \
                --dataset ${dataset} \
                --bias_type ${bias_type} \
                --y_key "y_helpfulness" \
                --z_key "y_reputation"


        bias_type="gender"

        experiment_id="make-natural-disjoint-${experiment}-${dataset}_m-${short_model_name_or_path}_i-${bias_type}"
        echo ${experiment_id}
        sbatch \
            -J ${experiment_id} \
            -o ${persistent_dir}/logs/data/%x.%j.out \
            -e ${persistent_dir}/logs/err/%x.%j.err \
            isambard.sh src/data/make_math_balanced_subset_natural_disjoint.py \
                --input_npz ${persistent_dir}/data/${experiment}/${dataset}/${model_name_or_path}/raw.npz \
                --output_npz ${persistent_dir}/data/${experiment}/${dataset}/${bias_type}/${model_name_or_path}/natural_disjoint.npz \
                --y_thr 3 --z_thr 185 \
                --n_clusters 5 \
                --split_seed 39 \
                --dataset ${dataset} \
                --bias_type ${bias_type} \
                --y_key "y_helpfulness" \
                --z_key "y_reputation"



        bias_type="location"

        experiment_id="make-natural-disjoint-${experiment}-${dataset}_m-${short_model_name_or_path}_i-${bias_type}"
        echo ${experiment_id}
        sbatch \
            -J ${experiment_id} \
            -o ${persistent_dir}/logs/data/%x.%j.out \
            -e ${persistent_dir}/logs/err/%x.%j.err \
            isambard.sh src/data/make_math_balanced_subset_natural_disjoint.py \
                --input_npz ${persistent_dir}/data/${experiment}/${dataset}/${model_name_or_path}/raw.npz \
                --output_npz ${persistent_dir}/data/${experiment}/${dataset}/${bias_type}/${model_name_or_path}/natural_disjoint.npz \
                --y_thr 3 --z_thr 185 \
                --n_clusters 5 \
                --split_seed 39 \
                --dataset ${dataset} \
                --bias_type ${bias_type} \
                --y_key "y_helpfulness" \
                --z_key "y_reputation"

    done
done



