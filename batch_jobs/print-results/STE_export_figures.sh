#!/bin/bash

source "batch_jobs/_experiment_configuration.sh"


# python src/export-results/STE_combined.py \
#   --persistent_dir . \
#   --model_name_or_path "bert-base-uncased|meta-llama/Llama-3.1-8B|mistralai/Mistral-7B-v0.3" \
#   --debiasing SAL \
#   --bias_type "reputation|location|engagement|name" \
#   --split "disjoint|joint|natural_disjoint|natural_joint" \
#   --out_dir results/summary_textual_erasure/STE_tables_clean



python src/export-results/STE_figure_all_models.py \
  --persistent_dir . \
  --model_name_or_path "bert-base-uncased|meta-llama/Llama-3.1-8B|mistralai/Mistral-7B-v0.3" \
  --debiasing SAL \
  --split "disjoint|joint|natural_disjoint|natural_joint" \
  --bias_type "gender" \
  --out_dir results/figures_by_model/

