#!/bin/bash

#SBATCH --gpus=1
#SBATCH --time=23:40:00         # Hours:Mins:Secs
#SBATCH -J so_dump
#SBATCH -o %x.%j.out
#SBATCH -e %x.%j.err

#SBATCH --mem=115G
#SBATCH --cpus-per-task=4

hostname
nvidia-smi --list-gpus

source ~/.bashrc
eval "$(/projects/u5gx/sshao/miniforge3/bin/conda shell.bash hook)"
conda activate msal
# hf auth login --token $HUGGINGFACE_TOKEN

# option 1: use curl with retry and resume support
curl -L -C - --fail --show-error --http1.1 \
  --retry 20 --retry-all-errors --retry-delay 30 \
  -o dump.7z \
  'https://data-dump.df1f831d6e65d11a023e73c38ab14ace.r2.cloudflarestorage.com/stackoverflow.com.7z?X-Amz-Expires=30&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=345b4a382d7759ee256c072cfbc12d36%2F20260127%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20260127T185819Z&X-Amz-SignedHeaders=host&X-Amz-Signature=f42f54634d7a9b60039b1a9005b2b15f3846c382a5fd656c626b5f7c11b44468'

# # or
# curl -L -C - --fail --show-error --http1.1 -o dump.7z "https://data-dump.df1f831d6e65d11a023e73c38ab14ace.r2.cloudflarestorage.com/stackoverflow.com.7z?X-Amz-Expires=30&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=345b4a382d7759ee256c072cfbc12d36%2F20260127%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20260127T223053Z&X-Amz-SignedHeaders=host&X-Amz-Signature=17854b7d6e312afd1494cdbf218e0b833d49db934cf721e6cb653458e75a9506"