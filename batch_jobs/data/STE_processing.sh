#!/bin/bash
set -euo pipefail

source "batch_jobs/_experiment_configuration.sh"

mkdir -p "${persistent_dir}/logs/data"
mkdir -p "${persistent_dir}/logs/err"

datasets=(
    "english"
    "superuser"
    "mathematics"
    "stackoverflow"
)


for dataset in "${datasets[@]}"; do
    experiment_id="clean-and-summarize-with-gender-${dataset}"
    echo "${experiment_id}"

    sbatch \
        -J "${experiment_id}" \
        -o "${persistent_dir}/logs/data/%x.%j.out" \
        -e "${persistent_dir}/logs/err/%x.%j.err" \
        isambard_cpu.sh \
        src/data/STE_process_and_summarize_with_gender.py \
            --input "data/STE/from_website/data/${dataset}.jsonl" \
            --cleaned_output "data/STE/raw/${dataset}/cleaned.jsonl" \
            --figdir "analysis/${dataset}" \
            --dataset_name "${dataset}" \
            --summary_output_jsonl "analysis/${dataset}/user_post_counts.jsonl" \
            --histogram_jsonl "analysis/${dataset}/user_post_count_histogram.jsonl" \
            --histogram_png "analysis/${dataset}/user_post_count_histogram.png" \
            --max_posts_per_user 200 \
            --bin_width 100
done