#!/bin/bash
source "batch_jobs/_experiment_configuration.sh"

mkdir -p ${persistent_dir}/logs_rebuttal/evaluation_bootstrap
mkdir -p ${persistent_dir}/logs_rebuttal/err
mkdir -p ${persistent_dir}/logs_rebuttal/test


all_models=(
    'BertModel'
    'LlamaModel'
    'MistralModel'
)


experiment="STE"
debiasing="SAL"
results_root="results_bootstrap"
bootstrap_samples=200
bootstrap_seed=0


for dataset in "mathematics" "stackoverflow" "english" "superuser"; do
    for split in "natural_disjoint"; do
        for Z_attribute in "textual" "random_textual" "binary"; do
            for bias_type in "gender" "location" "reputation" "engagement" "name"; do

                if [[ "$Z_attribute" == "textual" || "$Z_attribute" == "random_textual" ]]; then
                    num_eigenvectors_to_remove=50
                else
                    case "$bias_type" in
                        gender|reputation|engagement)
                            num_eigenvectors_to_remove=2
                            ;;
                        location|name)
                            num_eigenvectors_to_remove=5
                            ;;
                    esac
                fi

                echo "$dataset $split $Z_attribute $bias_type $num_eigenvectors_to_remove"



                for model in ${all_models[@]}; do
                    IFS=' ' read -r -a model_names <<< "${model_to_model_name_or_path[$model]}"
                    for model_name_or_path in "${model_names[@]}"; do
                        short_model_name_or_path=${model_name_or_path_to_short[${model_name_or_path}]}
                        experiment_id="bootstrap-eval-self-description-${experiment}-${dataset}-${split}_z-${Z_attribute}_d-${debiasing}_u-${num_eigenvectors_to_remove}_i-${bias_type}_m-${short_model_name_or_path}"

                        out_dir="${results_root}/${experiment}-${dataset}-${split}-${Z_attribute}/U_${num_eigenvectors_to_remove}/${model_name_or_path}/${debiasing}/${bias_type}"
                        pattern="${out_dir}"/*"${bias_type}".json

                        # if any matching file exists, skip
                        if compgen -G "$pattern" > /dev/null; then
                            echo "[SKIP] found existing: $pattern"
                            continue
                        else

                            if [[ "$split" == "joint" || "$split" == "disjoint" ]] && [[ "$bias_type" == "name" ]]; then
                                continue
                            fi

                            npz_path="data/${experiment}/${dataset}/${bias_type}/${model_name_or_path}/${split}.npz"


                            projection_matrix="model/${experiment}-${dataset}-${split}-${Z_attribute}/U_${num_eigenvectors_to_remove}/${model_name_or_path}/${debiasing}/${bias_type}.npz"

                            echo ${experiment_id}
                            sbatch \
                                -J ${experiment_id} \
                                -o ${persistent_dir}/logs_rebuttal/evaluation/%x.%j.out \
                                -e ${persistent_dir}/logs_rebuttal/err/%x.%j.err \
                                isambard.sh src/experiments/STE_evaluation_bootstrap.py \
                                    --persistent_dir ${persistent_dir} \
                                    --experiment ${experiment} \
                                    --model ${model} \
                                    --model_name_or_path ${model_name_or_path} \
                                    --X_attribute "input" \
                                    --debiasing ${debiasing} \
                                    --projection_matrix ${projection_matrix} \
                                    --bias_type ${bias_type} \
                                    --npz_path ${npz_path} \
                                    --results_save_path ${results_root}/${experiment}-${dataset}-${split}-${Z_attribute}/U_${num_eigenvectors_to_remove}/${model_name_or_path}/${debiasing}/${bias_type}/ \
                                    --num_eigenvectors_to_remove ${num_eigenvectors_to_remove} \
                                    --bootstrap_samples ${bootstrap_samples} \
                                    --bootstrap_seed ${bootstrap_seed}

                        fi
                    done
                done
            done
        done
    done
done

