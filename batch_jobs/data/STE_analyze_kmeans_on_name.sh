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

candidate_ks="2, 4, 6, 8, 10, 12, 16, 24, 32, 48, 64"


for model in ${all_models[@]}; do
    IFS=' ' read -r -a model_names <<< "${model_to_model_name_or_path[$model]}"
    for model_name_or_path in "${model_names[@]}"; do
        short_model_name_or_path=${model_name_or_path_to_short[${model_name_or_path}]}

        dataset="english"
        
        experiment_id="analyze-kmeans-disjoint-${experiment}-${dataset}_m-${short_model_name_or_path}_i-${bias_type}"
        echo ${experiment_id}
        sbatch \
            -J ${experiment_id} \
            -o ${persistent_dir}/logs/analysis/%x.%j.out \
            -e ${persistent_dir}/logs/err/%x.%j.err \
            isambard.sh src/data/analyze_kmeans_natural_disjoint.py \
                --input_npz ${persistent_dir}/data/${experiment}/${dataset}/${model_name_or_path}/raw.npz \
                --out_xlsx ${persistent_dir}/results/analysis/kmeans-disjoint-${experiment}-${dataset}-${bias_type}/${short_model_name_or_path}.xlsx \
                --y_thr 2 --z_thr 92133 \
                --split_seed 22 \
                --candidate_ks "${candidate_ks}" \
                --y_key "y_helpfulness" \
                --z_key "y_reputation"


        dataset="superuser"

        experiment_id="analyze-kmeans-joint-${experiment}-${dataset}_m-${short_model_name_or_path}_i-${bias_type}"
        echo ${experiment_id}
        sbatch \
            -J ${experiment_id} \
            -o ${persistent_dir}/logs/analysis/%x.%j.out \
            -e ${persistent_dir}/logs/err/%x.%j.err \
            isambard.sh src/data/analyze_kmeans_natural_disjoint.py \
                --input_npz ${persistent_dir}/data/${experiment}/${dataset}/${model_name_or_path}/raw.npz \
                --out_xlsx ${persistent_dir}/results/analysis/kmeans-joint-${experiment}-${dataset}-${bias_type}/${short_model_name_or_path}.xlsx \
                --y_thr 2 --z_thr 13711 \
                --candidate_ks "${candidate_ks}" \
                --y_key "y_helpfulness" \
                --z_key "y_reputation"

        dataset="stackoverflow"
        
        experiment_id="analyze-kmeans-joint-${experiment}-${dataset}_m-${short_model_name_or_path}_i-${bias_type}"
        echo ${experiment_id}
        sbatch \
            -J ${experiment_id} \
            -o ${persistent_dir}/logs/analysis/%x.%j.out \
            -e ${persistent_dir}/logs/err/%x.%j.err \
            isambard.sh src/data/analyze_kmeans_natural_disjoint.py \
                --input_npz ${persistent_dir}/data/${experiment}/${dataset}/${model_name_or_path}/raw.npz \
                --out_xlsx ${persistent_dir}/results/analysis/kmeans-joint-${experiment}-${dataset}-${bias_type}/${short_model_name_or_path}.xlsx \
                --y_thr 1 --z_thr 1752 \
                --candidate_ks "${candidate_ks}" \
                --y_key "y_helpfulness" \
                --z_key "y_reputation"

        
        dataset="mathematics"

        experiment_id="analyze-kmeans-joint-${experiment}-${dataset}_m-${short_model_name_or_path}_i-${bias_type}"
        echo ${experiment_id}
        sbatch \
            -J ${experiment_id} \
            -o ${persistent_dir}/logs/analysis/%x.%j.out \
            -e ${persistent_dir}/logs/err/%x.%j.err \
            isambard.sh src/data/analyze_kmeans_natural_disjoint.py \
                --input_npz ${persistent_dir}/data/${experiment}/${dataset}/${model_name_or_path}/raw.npz \
                --out_xlsx ${persistent_dir}/results/analysis/kmeans-joint-${experiment}-${dataset}-${bias_type}/${short_model_name_or_path}.xlsx \
                --y_thr 3 --z_thr 473652 \
                --candidate_ks "${candidate_ks}" \
                --y_key "y_helpfulness" \
                --z_key "y_reputation"

    done
done





