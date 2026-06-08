#!/bin/bash
set -euo pipefail

source "batch_jobs/_experiment_configuration.sh"

mkdir -p results/tables/data_distribution

experiment="STE"
split="natural_disjoint"

# Pass exactly one model key at a time. Default: BertModel.
# Example: bash STE_data_distribution_sal.sh BertModel
model="${1:-BertModel}"

# Read the actual model_name_or_path from your existing config.
# In your config:
#   BertModel -> bert-base-uncased
IFS=' ' read -r -a model_names <<< "${model_to_model_name_or_path[$model]}"
model_name_or_path="${model_names[0]}"

echo "Using model key: ${model}"
echo "Using model path: ${model_name_or_path}"

python src/data/STE_data_distribution.py \
    --data_root data \
    --experiment "${experiment}" \
    --datasets stackoverflow,mathematics,superuser,english \
    --bias_types reputation,gender,location,engagement,name \
    --model_name_or_path "${model_name_or_path}" \
    --split_file "${split}" \
    --output_dir results/tables/data_distribution