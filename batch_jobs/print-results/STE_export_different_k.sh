#!/bin/bash

source "batch_jobs/_experiment_configuration.sh"


python src/export-results/STE_summary.py \
  --persistent_dir . \
  --model_name_or_path "bert-base-uncased|meta-llama/Llama-3.1-8B|mistralai/Mistral-7B-v0.3" \
  --debiasing SAL \
  --split "natural_disjoint" \
  --k_random_name 50 \
  --k_random_location 50 \
  --k_random_reputation 50 \
  --k_random_engagement 50 \
  --k_random_gender 50 \
  --k_explicit_name 5 \
  --k_explicit_location 5 \
  --k_explicit_reputation 2 \
  --k_explicit_engagement 2 \
  --k_explicit_gender 2 \
  --k_implicit_name 2 \
  --k_implicit_location 2 \
  --k_implicit_reputation 2 \
  --k_implicit_engagement 2 \
  --k_implicit_gender 2 \
  --out_dir results/summary_textual_erasure/STE_tables_summary_implicit-2


python src/export-results/STE_summary.py \
  --persistent_dir . \
  --model_name_or_path "bert-base-uncased|meta-llama/Llama-3.1-8B|mistralai/Mistral-7B-v0.3" \
  --debiasing SAL \
  --split "natural_disjoint" \
  --k_random_name 5 \
  --k_random_location 5 \
  --k_random_reputation 5 \
  --k_random_engagement 5 \
  --k_random_gender 5 \
  --k_explicit_name 5 \
  --k_explicit_location 5 \
  --k_explicit_reputation 2 \
  --k_explicit_engagement 2 \
  --k_explicit_gender 2 \
  --k_implicit_name 5 \
  --k_implicit_location 5 \
  --k_implicit_reputation 5 \
  --k_implicit_engagement 5 \
  --k_implicit_gender 5 \
  --out_dir results/summary_textual_erasure/STE_tables_summary_implicit-5



python src/export-results/STE_summary.py \
  --persistent_dir . \
  --model_name_or_path "bert-base-uncased|meta-llama/Llama-3.1-8B|mistralai/Mistral-7B-v0.3" \
  --debiasing SAL \
  --split "natural_disjoint" \
  --k_random_name 10 \
  --k_random_location 10 \
  --k_random_reputation 10 \
  --k_random_engagement 10 \
  --k_random_gender 10 \
  --k_explicit_name 5 \
  --k_explicit_location 5 \
  --k_explicit_reputation 2 \
  --k_explicit_engagement 2 \
  --k_explicit_gender 2 \
  --k_implicit_name 10 \
  --k_implicit_location 10 \
  --k_implicit_reputation 10 \
  --k_implicit_engagement 10 \
  --k_implicit_gender 10 \
  --out_dir results/summary_textual_erasure/STE_tables_summary_implicit-10




python src/export-results/STE_summary.py \
  --persistent_dir . \
  --model_name_or_path "bert-base-uncased|meta-llama/Llama-3.1-8B|mistralai/Mistral-7B-v0.3" \
  --debiasing SAL \
  --split "natural_disjoint" \
  --k_random_name 24 \
  --k_random_location 24 \
  --k_random_reputation 24 \
  --k_random_engagement 24 \
  --k_random_gender 24 \
  --k_explicit_name 5 \
  --k_explicit_location 5 \
  --k_explicit_reputation 2 \
  --k_explicit_engagement 2 \
  --k_explicit_gender 2 \
  --k_implicit_name 24 \
  --k_implicit_location 24 \
  --k_implicit_reputation 24 \
  --k_implicit_engagement 24 \
  --k_implicit_gender 24 \
  --out_dir results/summary_textual_erasure/STE_tables_summary_implicit-24


python src/export-results/STE_summary.py \
  --persistent_dir . \
  --model_name_or_path "bert-base-uncased|meta-llama/Llama-3.1-8B|mistralai/Mistral-7B-v0.3" \
  --debiasing SAL \
  --split "natural_disjoint" \
  --k_random_name 45 \
  --k_random_location 45 \
  --k_random_reputation 45 \
  --k_random_engagement 45 \
  --k_random_gender 45 \
  --k_explicit_name 5 \
  --k_explicit_location 5 \
  --k_explicit_reputation 2 \
  --k_explicit_engagement 2 \
  --k_explicit_gender 2 \
  --k_implicit_name 45 \
  --k_implicit_location 45 \
  --k_implicit_reputation 45 \
  --k_implicit_engagement 45 \
  --k_implicit_gender 45 \
  --out_dir results/summary_textual_erasure/STE_tables_summary_implicit-45


python src/export-results/STE_summary.py \
  --persistent_dir . \
  --model_name_or_path "bert-base-uncased|meta-llama/Llama-3.1-8B|mistralai/Mistral-7B-v0.3" \
  --debiasing SAL \
  --split "natural_disjoint" \
  --k_random_name 50 \
  --k_random_location 50 \
  --k_random_reputation 50 \
  --k_random_engagement 50 \
  --k_random_gender 50 \
  --k_explicit_name 5 \
  --k_explicit_location 5 \
  --k_explicit_reputation 2 \
  --k_explicit_engagement 2 \
  --k_explicit_gender 2 \
  --k_implicit_name 50 \
  --k_implicit_location 50 \
  --k_implicit_reputation 50 \
  --k_implicit_engagement 50 \
  --k_implicit_gender 50 \
  --out_dir results/summary_textual_erasure/STE_tables_summary_implicit-50
