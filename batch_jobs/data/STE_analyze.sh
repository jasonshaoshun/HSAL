#!/bin/bash
set -euo pipefail

source "batch_jobs/_experiment_configuration.sh"

mkdir -p ${persistent_dir}/logs/data
mkdir -p ${persistent_dir}/logs/err

all_models=(
    'BertModel'
    'LlamaModel'
    'MistralModel'
)

datasets=(
    "english"
    "mathematics"
    "stackoverflow"
    "superuser"
)

experiment="STE"

# Use the IceLake runner by default. Override with:
#   CPU_RUNNER=csd3_cclake.sh bash STE_analyze_parallel.sh
CPU_RUNNER="${CPU_RUNNER:-isambard_cpu.sh}"

# Outer parallelism over threshold pairs. On one IceLake node, 32 is a good first value.
# Increase to 48/64/76 if memory and CPU efficiency look good.
PARALLEL_JOBS="${PARALLEL_JOBS:-4}"

for dataset in ${datasets[@]}; do
    for bias_type in "y_engagement" "y_reputation"; do
        for model in ${all_models[@]}; do
            IFS=' ' read -r -a model_names <<< "${model_to_model_name_or_path[$model]}"
            for model_name_or_path in "${model_names[@]}"; do
                short_model_name_or_path=${model_name_or_path_to_short[${model_name_or_path}]}

                experiment_id="analyze-zbalanced-natural-disjoint_${experiment}-${dataset}_m-${short_model_name_or_path}_i-${bias_type}_filter"
                echo ${experiment_id}
                sbatch \
                    -J ${experiment_id} \
                    -o ${persistent_dir}/logs/data/%x.%j.out \
                    -e ${persistent_dir}/logs/err/%x.%j.err \
                    ${CPU_RUNNER} src/data/find_good_thresholds_zbalanced_disjoint_parallel.py \
                        --input_npz ${persistent_dir}/data/${experiment}/${dataset}/${model_name_or_path}/raw.npz \
                        --out_xlsx ${persistent_dir}/results/analysis/zbalanced-disjoint-${experiment}-${dataset}-${bias_type}/natural-disjoint-${short_model_name_or_path}.xlsx \
                        --y_key y_helpfulness --z_key ${bias_type} \
                        --parallel_jobs ${PARALLEL_JOBS} \
                        --lr_n_jobs 1
            done
        done
    done
done



# # For single CPU runs, use:

# datasets=(
#     'mathematics'
#     'stackoverflow'
#     'english'
#     'superuser'
# )


# for dataset in ${datasets[@]}; do
#     for bias_type in "y_engagement" "y_reputation"; do
#         for model in ${all_models[@]}; do
#             IFS=' ' read -r -a model_names <<< "${model_to_model_name_or_path[$model]}"
#             for model_name_or_path in "${model_names[@]}"; do
#                 short_model_name_or_path=${model_name_or_path_to_short[${model_name_or_path}]}

#                 experiment_id="analyze-zbalanced-natural-disjoint_${experiment}-${dataset}_m-${short_model_name_or_path}_i-${bias_type}_filter"
#                 echo ${experiment_id}
#                 sbatch \
#                     -J ${experiment_id} \
#                     -o ${persistent_dir}/logs/data/%x.%j.out \
#                     -e ${persistent_dir}/logs/err/%x.%j.err \
#                     isambard.sh src/data/find_good_thresholds_zbalanced_disjoint.py \
#                         --input_npz ${persistent_dir}/data/${experiment}/${dataset}/${model_name_or_path}/raw.npz \
#                         --out_xlsx ${persistent_dir}/results/analysis/zbalanced-disjoint-${experiment}-${dataset}-${bias_type}/natural-disjoint-${short_model_name_or_path}.xlsx \
#                         --y_key y_helpfulness --z_key ${bias_type}
                
#             done
#         done
#     done
# done


