# Textual Erasure

<!-- This repository contains the code for preparing and processing data, and replicating the results in the paper "Iterative Multilingual Spectral Attribute Erasure". -->

## Create the Environment

### Option 1. Create Environment Manually

```sh
# Create a new conda environment with Python:
conda create -n msal python=3.10
# activate the environment
conda activate msal
# Install required packages one by one:
pip install -r requirements.txt
```

Install pytorch with CUDA:
```sh
# cuda 12.1
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

### Option 2. Or Create Environment Automatically

```sh
# Create the environment using the file:
conda env create -f environment.yml
```

## Project Overview

This repository builds a Stack Exchange fairness benchmark and evaluates textual erasure with H-SAL. Each example contains:

- `X`: the question-answer text used for helpfulness prediction.
- `H`: the user's free-form self-description text from `AboutMe`.
- `Y`: answer helpfulness, derived from `AnswerScore`.
- `Z`: user attributes used for fairness evaluation, including gender, country/location, display-name cluster, reputation, and engagement.

H-SAL estimates an erasure subspace from profile-text representations rather than directly using the protected attribute labels. Standard SAL, random-direction removal, and no-erasure baselines are used for comparison.

## Configure Paths

Before launching batch jobs, edit [batch_jobs/_experiment_configuration.sh](batch_jobs/_experiment_configuration.sh) and set:

- `persistent_dir`: project root used for data, logs, models, and results.
- `model_name_or_path` and `model_to_model_name_or_path`: backbone models to run.
- `model_name_or_path_to_short`: readable names used in job IDs and output filenames.

The scripts are written for Slurm and use [isambard.sh](isambard.sh) or [isambard_cpu.sh](isambard_cpu.sh). Update conda paths, queue settings, memory, GPU count, and `HUGGINGFACE_TOKEN` handling for your cluster.

## Data Dumping

Download the public Stack Exchange `.7z` dumps, then place either archives or extracted folders under:

```text
data/STE/from_website/dumps/
```

You can get `[URL]` from the Stack Exchange data-dump download page for the community you want. The signed URLs expire, so the repository documents the command shape rather than storing one fixed URL:

```sh
# option 1: use curl with retry and resume support
curl -L -C - --fail --show-error --http1.1 \
  --retry 20 --retry-all-errors --retry-delay 30 \
  -o dump.7z \
  '[URL]'

# or
curl -L -C - --fail --show-error --http1.1 -o dump.7z "[URL]"
```

Expected archive or folder names are:

```text
english.stackexchange.com.7z      or english/
math.stackexchange.com.7z         or mathematics/
stackoverflow.com.7z              or stackoverflow/
superuser.com.7z                  or superuser/
```

Convert the XML dumps into the raw JSONL files expected by the STE pipeline:

```sh
# Convert all supported datasets locally.
bash batch_jobs/data/STE_dump_stackexchange.sh

# Convert one dataset.
DATASETS="superuser" bash batch_jobs/data/STE_dump_stackexchange.sh

# Submit one Slurm job per dataset.
SUBMIT=1 DATASETS="english mathematics stackoverflow superuser" bash batch_jobs/data/STE_dump_stackexchange.sh
```

The wrapper calls [src/data/STE_extract_stackexchange_dump.py](src/data/STE_extract_stackexchange_dump.py), which replaces the old Super User-only converter with a reusable extractor:

```sh
python -u src/data/STE_extract_stackexchange_dump.py \
  --dataset superuser \
  --input-dir data/STE/from_website/dumps/superuser \
  --output data/STE/from_website/data/superuser.jsonl
```

The generated raw files should be:

```text
data/STE/from_website/data/english.jsonl
data/STE/from_website/data/mathematics.jsonl
data/STE/from_website/data/stackoverflow.jsonl
data/STE/from_website/data/superuser.jsonl
```

## Data Pipeline

Run the data jobs in this order.

### 1. Clean and Summarize

```sh
bash batch_jobs/data/STE_processing.sh
```

This reads `data/STE/from_website/data/<dataset>.jsonl`, strips HTML/code, builds `TextClean` and `AboutMeClean`, normalizes country labels, derives gender from display names, caps posts per user, and writes:

```text
data/STE/raw/<dataset>/cleaned.jsonl
analysis/<dataset>/
```

### 2. Encode Text

```sh
bash batch_jobs/data/STE_encode.sh
```

This encodes question-answer text, user self-descriptions, and display names for each configured backbone model. Outputs are written as:

```text
data/STE/<dataset>/<model_name_or_path>/raw.npz
```

### 3. Analyze Thresholds

```sh
bash batch_jobs/data/STE_analyze.sh
```

This searches threshold grids for user-disjoint, Z-balanced settings and writes spreadsheets under:

```text
results/analysis/zbalanced-disjoint-STE-<dataset>-<bias_type>/
```

Use these spreadsheets to choose dataset-specific thresholds.

### 4. Build Natural User-Disjoint Splits

After selecting thresholds, create the final datasets:

```sh
bash batch_jobs/data/STE_filter_english.sh
bash batch_jobs/data/STE_filter_mathematics.sh
bash batch_jobs/data/STE_filter_stackoverflow.sh
bash batch_jobs/data/STE_filter_superuser.sh
```

The current selected thresholds in these scripts are:

| Dataset | Helpfulness `y_thr` | Reputation `z_thr` | Engagement `z_thr` |
| --- | ---: | ---: | ---: |
| english | 3 | 185 | 6 |
| mathematics | 2 | 466 | 22 |
| stackoverflow | 2 | 2498 | 246 |
| superuser | 2 | 121 | 2 |

The split files are written as:

```text
data/STE/<dataset>/<bias_type>/<model_name_or_path>/natural_disjoint.npz
```

## Debiasing and Evaluation

Compute SAL or H-SAL projection matrices:

```sh
bash batch_jobs/STE_compute_sal.sh
```

Evaluate helpfulness prediction and TPR-Gap after projection:

```sh
bash batch_jobs/STE_evaluation_sal.sh
```

The main paper uses the natural user-disjoint split across BERT, Llama-3.1-8B, and Mistral-7B. The batch scripts may be narrowed for a smaller run; expand the loops over `Z_attribute`, `bias_type`, and `num_eigenvectors_to_remove` in the compute/evaluation scripts to reproduce the full set of settings. In particular:

- `Z_attribute=binary`: explicit SAL using protected attribute labels.
- `Z_attribute=textual`: H-SAL using self-description representations.
- `Z_attribute=random_textual`: random-direction control.
- `bias_type`: `gender`, `location`, `reputation`, `engagement`, or `name`.

## Attribute Predictability Ablation

To test whether `Z` is predictable from embeddings under the user-disjoint split, run the logistic-regression probing ablation:

```sh
bash batch_jobs/STE_eval_predict_z_from_x_raw_unique_logreg.sh
```

This submits one probing job per dataset, model, and bias type, then exports:

```text
results/tables/predict_z_from_x_natural_disjoint.tex
```

## Result Export

After evaluation, use the print/export scripts to build figures and different-k summaries:

```sh
bash batch_jobs/print-results/STE_export_figures.sh
bash batch_jobs/print-results/STE_export_different_k.sh
```

For singular value plots:

```sh
python src/experiments/plot_singular_values.py \
  --values_txt data/singular_values.txt \
  --out_dir figures/singular_values \
  --logy
```

## Repository Map

```text
batch_jobs/                  Slurm launchers for data, debiasing, evaluation, and export
batch_jobs/data/             Data dumping, processing, encoding, thresholding, and filtering jobs
src/data/                    Dump extraction, cleaning, encoding, threshold search, split creation
src/experiments/             SAL/H-SAL computation, evaluation, and probing experiments
src/debias/                  Debiasing implementations
src/model/                   Model wrappers
src/export-results/          Table and figure export scripts
analysis/                    Dataset diagnostics generated during processing
results/                     Evaluation outputs and exported tables
```
