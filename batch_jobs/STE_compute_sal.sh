#!/bin/bash

source "batch_jobs/_experiment_configuration.sh"

mkdir -p ${persistent_dir}/logs/compute
mkdir -p ${persistent_dir}/logs/err

all_models=(
    'BertModel'
    'LlamaModel'
    'MistralModel'
)

debiasing="SAL"
experiment="STE"


for dataset in "mathematics" "stackoverflow" "english" "superuser"; do
    for split in "natural_disjoint"; do
        # for Z_attribute in "textual" "random_textual" "binary"; do
        for Z_attribute in "random_textual" "binary" ; do

            for bias_type in "gender" "location" "reputation" "engagement" "name"; do
            # for bias_type in "gender"; do
                # for num_eigenvectors_to_remove in 1 2 4 5 6 10 24 25 45 50; do
                for num_eigenvectors_to_remove in 50; do



                    for model in ${all_models[@]}; do
                        IFS=' ' read -r -a model_names <<< "${model_to_model_name_or_path[$model]}"

                        for model_name_or_path in "${model_names[@]}"; do

                            out_dir="model/${experiment}-${dataset}-${split}-${Z_attribute}/U_${num_eigenvectors_to_remove}/${model_name_or_path}/${debiasing}"
                            pattern="${out_dir}"/*"${bias_type}".npz

                            # if any matching file exists, skip
                            if compgen -G "$pattern" > /dev/null; then
                                echo "[SKIP] found existing: $pattern"
                                continue
                            else

                                if [[ "$split" == "joint" || "$split" == "disjoint" ]] && [[ "$bias_type" == "name" ]]; then
                                    continue
                                fi

                                # if [[ "$bias_type" == "location" ]]; then
                                #     npz_path="data/${experiment}/${dataset}/reputation/${model_name_or_path}/${split}.npz"
                                # else
                                #     npz_path="data/${experiment}/${dataset}/${bias_type}/${model_name_or_path}/${split}.npz"
                                # fi

                                npz_path="data/${experiment}/${dataset}/${bias_type}/${model_name_or_path}/${split}.npz"
                                
                                # if [[ "$Z_attribute" == "random_textual" || "$Z_attribute" == "textual" ]] && [[ "$bias_type" == "location" ]]; then
                                #     continue
                                # fi

                                short_model_name_or_path=${model_name_or_path_to_short[${model_name_or_path}]}
                                experiment_id="compute-sal-${experiment}-${dataset}-${split}_z-${Z_attribute}_u-${num_eigenvectors_to_remove}_m-${short_model_name_or_path}_i-${bias_type}"
                                echo ${experiment_id}
                                sbatch \
                                    -J ${experiment_id} \
                                    -o ${persistent_dir}/logs/compute/%x.%j.out \
                                    -e ${persistent_dir}/logs/err/%x.%j.err \
                                    isambard.sh src/experiments/STE_compute_sal.py \
                                        --persistent_dir ${persistent_dir} \
                                        --model ${model} \
                                        --model_name_or_path ${model_name_or_path} \
                                        --bias_type ${bias_type} \
                                        --dataset_path ${npz_path} \
                                        --save_path "model/${experiment}-${dataset}-${split}-${Z_attribute}/U_${num_eigenvectors_to_remove}/${model_name_or_path}/${debiasing}/" \
                                        --num_eigenvectors_to_remove ${num_eigenvectors_to_remove} \
                                        --Z_attribute ${Z_attribute}


                            fi
                        done
                    done
                done
            done
        done
    done
done

