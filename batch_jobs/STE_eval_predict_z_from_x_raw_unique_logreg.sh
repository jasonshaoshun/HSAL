#!/bin/bash
source "batch_jobs/_experiment_configuration.sh"

mkdir -p ${persistent_dir}/logs/analysis_logreg

all_models=(
    'BertModel'
    'LlamaModel'
    'MistralModel'
)

experiment="STE"
split="natural_disjoint"
results_root="results/analysis_on_Z_aboutme_logreg"



for dataset in "english" "mathematics" "stackoverflow" "superuser"; do
    for bias_type in "reputation" "gender" "location" "engagement"; do
        for model in ${all_models[@]}; do
            IFS=' ' read -r -a model_names <<< "${model_to_model_name_or_path[$model]}"
            for model_name_or_path in "${model_names[@]}"; do
                short_model_name_or_path=${model_name_or_path_to_short[${model_name_or_path}]}
                experiment_id="predict-z-from-aboutme-cls-${experiment}-${dataset}-${bias_type}-${short_model_name_or_path}-logreg"

                npz_path="data/${experiment}/${dataset}/${bias_type}/${model_name_or_path}/natural_disjoint.npz"

                results_dir="${results_root}/${experiment}-${dataset}-${split}/${model_name_or_path}/${bias_type}"
                mkdir -p "${results_dir}"

                echo "Submitting: ${experiment_id} bias_type=${bias_type}"
                echo "  npz_path: ${npz_path}"
                echo "  results_dir: ${results_dir}"
                echo "  classifier search: 5 logistic_regression"

                sbatch \
                    -J ${experiment_id} \
                    -o ${persistent_dir}/logs/analysis_logreg/%x.%j.out \
                    -e ${persistent_dir}/logs/analysis_logreg/%x.%j.err \
                    isambard.sh src/experiments/STE_eval_predict_z_from_x_unique.py \
                        --experiment ${experiment} \
                        --dataset ${dataset} \
                        --model ${model} \
                        --model_name_or_path ${model_name_or_path} \
                        --bias_type ${bias_type} \
                        --npz_path ${npz_path} \
                        --results_save_path ${results_dir}

            done
        done
    done
done




python src/export-results/build_table_predict_z_from_x_raw_unique.py \
    --results_root ${results_root} \
    --experiment ${experiment} \
    --split ${split} \
    --prefer logreg \
    --output_tex results/tables/predict_z_from_x_natural_disjoint.tex