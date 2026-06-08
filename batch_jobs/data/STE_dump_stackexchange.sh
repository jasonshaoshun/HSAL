#!/bin/bash
set -euo pipefail

if [[ -f "batch_jobs/_experiment_configuration.sh" ]]; then
    source "batch_jobs/_experiment_configuration.sh"
fi

DATASETS_INPUT="${DATASETS:-english superuser mathematics stackoverflow}"
read -r -a datasets <<< "${DATASETS_INPUT}"

DUMP_ROOT="${DUMP_ROOT:-data/STE/from_website/dumps}"
OUTPUT_DIR="${OUTPUT_DIR:-data/STE/from_website/data}"
SUBMIT="${SUBMIT:-0}"
PYTHON="${PYTHON:-python}"
CPU_RUNNER="${CPU_RUNNER:-isambard_cpu.sh}"
LOG_DIR="${LOG_DIR:-${persistent_dir:-.}/logs/data}"
ERR_DIR="${ERR_DIR:-${persistent_dir:-.}/logs/err}"
read -r -a python_cmd <<< "${PYTHON}"

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}" "${ERR_DIR}"

declare -A site_by_dataset=(
    ["english"]="english.stackexchange.com"
    ["mathematics"]="math.stackexchange.com"
    ["stackoverflow"]="stackoverflow.com"
    ["superuser"]="superuser.com"
)

for dataset in "${datasets[@]}"; do
    site="${site_by_dataset[$dataset]:-$dataset}"

    input_dir=""
    for candidate in "${DUMP_ROOT}/${dataset}" "${DUMP_ROOT}/${site}"; do
        if [[ -d "${candidate}" ]]; then
            input_dir="${candidate}"
            break
        fi
    done

    archive=""
    for candidate in "${DUMP_ROOT}/${dataset}.7z" "${DUMP_ROOT}/${site}.7z"; do
        if [[ -f "${candidate}" ]]; then
            archive="${candidate}"
            break
        fi
    done

    args=(
        src/data/STE_extract_stackexchange_dump.py
        --dataset "${dataset}"
        --site "${site}"
        --output "${OUTPUT_DIR}/${dataset}.jsonl"
    )

    if [[ -n "${archive}" ]]; then
        args+=(--archive "${archive}" --extract-dir "${DUMP_ROOT}/${dataset}")
    elif [[ -n "${input_dir}" ]]; then
        args+=(--input-dir "${input_dir}")
    else
        echo "Missing dump for dataset=${dataset}." >&2
        echo "Expected an extracted directory at ${DUMP_ROOT}/${dataset}/ or ${DUMP_ROOT}/${site}/," >&2
        echo "or an archive at ${DUMP_ROOT}/${dataset}.7z or ${DUMP_ROOT}/${site}.7z." >&2
        exit 1
    fi

    if [[ -n "${START_DATE:-}" ]]; then
        args+=(--start-date "${START_DATE}")
    fi
    if [[ -n "${END_DATE:-}" ]]; then
        args+=(--end-date "${END_DATE}")
    fi
    if [[ -n "${MAX_ROWS:-}" ]]; then
        args+=(--max-rows "${MAX_ROWS}")
    fi
    if [[ "${KEEP_CODE_POSTS:-0}" == "1" ]]; then
        args+=(--keep-code-posts)
    fi
    if [[ "${INCLUDE_COMMUNITY_OWNED:-0}" == "1" ]]; then
        args+=(--include-community-owned)
    fi

    experiment_id="dump-stackexchange-${dataset}"
    echo "${experiment_id}"

    if [[ "${SUBMIT}" == "1" ]]; then
        sbatch \
            -J "${experiment_id}" \
            -o "${LOG_DIR}/%x.%j.out" \
            -e "${ERR_DIR}/%x.%j.err" \
            "${CPU_RUNNER}" "${args[@]}"
    else
        "${python_cmd[@]}" -u "${args[@]}"
    fi
done
