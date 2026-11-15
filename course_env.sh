#!/bin/bash
# ==============================================================================
# Course Configuration & Environment Settings
# LLM Fine-Tuning & Inference on HPC (Olivia Supercomputer)
# ==============================================================================

# 1. Slurm Account & Reservation
# If the course has an active reservation, set COURSE_SLURM_RESERVATION="reservation_name"
export COURSE_SLURM_ACCOUNT="${COURSE_SLURM_ACCOUNT:-nn9997k}"
export COURSE_SLURM_RESERVATION="${COURSE_SLURM_RESERVATION:-}"

# 2. Personal & Cluster Workspace Setup
export CLUSTER_PROJECT_ROOT="${CLUSTER_PROJECT_ROOT:-/cluster/projects/${COURSE_SLURM_ACCOUNT}}"
export PROJECT_DIR="${PROJECT_DIR:-${CLUSTER_PROJECT_ROOT}/${USER}/llm-hpc-course}"
export SHARED_DIR="${SHARED_DIR:-${PROJECT_DIR}/shared}"
export RESULT_DIR="${RESULT_DIR:-${PROJECT_DIR}/results}"

# 3. Model & Dataset Components
export BASE_MODEL_DIR="${BASE_MODEL_DIR:-${SHARED_DIR}/models/Llama-3.2-1B-Instruct}"
export TOKENIZER_DIR="${TOKENIZER_DIR:-${BASE_MODEL_DIR}/original/tokenizer.model}"
export QA_DATASET_DIR="${QA_DATASET_DIR:-${SHARED_DIR}/datasets/GPT-4-LLM/alpaca_gpt4_data.json}"

# 4. Apptainer Container Configuration
export CONTAINER_DIR="${CONTAINER_DIR:-${PROJECT_DIR}/apptainer}"
export APPTAINER_SIF="${APPTAINER_SIF:-${CONTAINER_DIR}/pytorch_25.08_cuda13.0_arm_custom.sif}"
export VLLM_SIF="${VLLM_SIF:-${CONTAINER_DIR}/vllm0.12_cu131_py3.12_arm_custom.sif}"

# 5. HuggingFace & vLLM Caching
export HF_HOME="${HF_HOME:-${PROJECT_DIR}/hf_cache}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-${PROJECT_DIR}/.cache/vllm}"

# 6. Default Execution Backend ("container" or "eessi")
export DEFAULT_BACKEND="${DEFAULT_BACKEND:-container}"

