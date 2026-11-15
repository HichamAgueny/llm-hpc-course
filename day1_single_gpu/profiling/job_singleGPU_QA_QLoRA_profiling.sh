#!/bin/bash -e
#SBATCH --job-name=ft-llama3-1B-qlora-1gpu_profiling
#SBATCH --account=nn9997k
#SBATCH --time=00:10:00
#SBATCH --partition=accel
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH -o ./out/%x-%j.out
#SBATCH --mem-per-gpu=96G
#SBATCH --reservation=software

echo "--Node: $(hostname)"
echo

export PYTHONNOUSERSITE=1

# =====================================================================
# --- CONFIGURABLE VARIABLES AND PATHS (USERS ONLY MODIFY THIS BLOCK) ---
# =====================================================================

# 1. Base Project Allocation (Change this if migrating to a different project share)
export CLUSTER_PROJECT_ROOT="/cluster/work/projects/nn9971k"

# 2. Personal Workspace Setup (Derived automatically from your root above)
export PROJECT_DIR="${CLUSTER_PROJECT_ROOT}/${USER}/llm-hpc-course"
export SHARED_DIR="${PROJECT_DIR}/shared"
export RESULT_DIR="${PROJECT_DIR}/results"

# 3. Input Component Paths
export CHECKPOINT_DIR="${SHARED_DIR}/models/Llama-3.2-1B-Instruct"
export TOKENIZER_DIR="${CHECKPOINT_DIR}/original/tokenizer.model"
export DATASET_DIR="${SHARED_DIR}/datasets/GPT-4-LLM/alpaca_gpt4_data.json"

# 4. Output Component Paths
export RUN_NAME="profiling_llama3.2-1b_qlora_single_device"
#export OUTPUT_DIR="${RESULT_DIR}/${RUN_NAME}/checkpoints"
export OUTPUT_DIR="/tmp/${USER}_${SLURM_JOB_ID}/checkpoints"
export LOG_DIR="${RESULT_DIR}/profiling_results/${RUN_NAME}/logs"
export PROFILING_DIR="${RESULT_DIR}/profiling_results/${RUN_NAME}/profiles"

# 5. Script / Config Layout
export CONFIG_FILE="${PROJECT_DIR}/configs/qlora/llama3_2_1B_qlora_single_device_profiling.yaml"
export PYTHON_FILE="${PROJECT_DIR}/recipes/single_device/lora_finetune_single_device.py"

# 6. Apptainer Container Configuration
export CONTAINER_DIR="${PROJECT_DIR}/apptainer"
export APPTAINER_SIF="${CONTAINER_DIR}/pytorch_25.08_cuda13.0_arm_custom.sif"

# =====================================================================

echo "--- My Main Directory (host): ${PROJECT_DIR}"
echo "--- Bind-mounted inside container as: /workspace"
echo
echo "=== Running inside Apptainer ==="
echo "CONFIG_FILE: ${CONFIG_FILE}"
echo "PYTHON_FILE: ${PYTHON_FILE}"
echo

echo "--- Launching the application inside Apptainer ---"

# --- Execute with Apptainer ---
time srun apptainer exec --nv \
      -B "${PROJECT_DIR}:/workspace" \
      -B "${CLUSTER_PROJECT_ROOT}" \
      "${APPTAINER_SIF}" \
      python "${PYTHON_FILE}" --config "${CONFIG_FILE}" \
        --checkpoint_dir "${CHECKPOINT_DIR}" \
        --tokenizer_dir "${TOKENIZER_DIR}" \
        --dataset_dir "${DATASET_DIR}" \
        --output_dir "${OUTPUT_DIR}" \
        --log_dir "${LOG_DIR}" \
        --profiling_dir "${PROFILING_DIR}"

echo
echo "--- Finished :) ---"
