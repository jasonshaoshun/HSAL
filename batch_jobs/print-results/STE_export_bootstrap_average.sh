#!/bin/bash

source "batch_jobs/_experiment_configuration.sh"

common_args=(
  --persistent_dir .
  --model_name_or_path "bert-base-uncased|meta-llama/Llama-3.1-8B|mistralai/Mistral-7B-v0.3"
  --debiasing SAL
  --split "natural_disjoint"
  --results_root results_bootstrap
  --k_random_name 50
  --k_random_location 50
  --k_random_reputation 50
  --k_random_engagement 50
  --k_random_gender 50
  --k_explicit_name 5
  --k_explicit_location 5
  --k_explicit_reputation 2
  --k_explicit_engagement 2
  --k_explicit_gender 2
  --k_implicit_name 50
  --k_implicit_location 50
  --k_implicit_reputation 50
  --k_implicit_engagement 50
  --k_implicit_gender 50
)

python src/export-results/STE_bootstrap_summary.py \
  "${common_args[@]}" \
  --out_dir results_bootstrap/summary_textual_erasure/STE_tables_bootstrap_summary_paper

python src/export-results/STE_bootstrap_summary_plain.py \
  "${common_args[@]}" \
  --out_dir results_bootstrap/summary_textual_erasure/STE_tables_bootstrap_summary_plain
