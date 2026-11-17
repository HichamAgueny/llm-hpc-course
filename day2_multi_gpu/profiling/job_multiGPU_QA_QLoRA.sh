#!/bin/bash -e
#SBATCH --job-name=ft-llama3-1B-qlora-4gpu_profiling
#SBATCH --account=nn9971k
#SBATCH --time=00:15:00
#SBATCH --partition=accel
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH -o ./out/%x-%j.out
#SBATCH --mem-per-gpu=97G
#SBATCH --reservation=nn9971k_213

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
export RUN_NAME="profiling_llama3.2-1b_qlora_multi_device"
# Throw-away checkpoints go to the node-local /tmp; profiles and logs are kept
export OUTPUT_DIR="/tmp/${USER}_${SLURM_JOB_ID}/checkpoints"
export LOG_DIR="${RESULT_DIR}/profiling_results/${RUN_NAME}/logs"
export PROFILING_DIR="${RESULT_DIR}/profiling_results/${RUN_NAME}/profiles"

# 5. Script / Config Layout
export CONFIG_FILE="${PROJECT_DIR}/configs/qlora/llama3_2_1B_qlora_multi_device_profiling.yaml"
export PYTHON_FILE="${PROJECT_DIR}/recipes/distributed/lora_finetune_distributed.py"

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

# Flash Attention for efficiency
export USE_FLASH_ATTENTION=1

# Set up variables to control distributed PyTorch training
export MASTER_ADDR=$(hostname)
export MASTER_PORT=25900
export WORLD_SIZE=$SLURM_NPROCS
export LOCAL_WORLD_SIZE=$SLURM_GPUS_PER_NODE

# --- Define the training task function ---
run_training() {

    # These resolve dynamically on each task/GPU node during srun
    export RANK=$SLURM_PROCID
    export LOCAL_RANK=$SLURM_LOCALID

    echo "Task ${SLURM_PROCID}: RANK=${RANK}, LOCAL_RANK=${LOCAL_RANK}, WORLD_SIZE=${WORLD_SIZE}, LOCAL_WORLD_SIZE=${LOCAL_WORLD_SIZE}"
    echo "LOCAL_RANK: ${LOCAL_RANK}, CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"

    python "${PYTHON_FILE}" --config "${CONFIG_FILE}" \
             --checkpoint_dir "${CHECKPOINT_DIR}" \
             --tokenizer_dir "${TOKENIZER_DIR}" \
             --dataset_dir "${DATASET_DIR}" \
             --output_dir "${OUTPUT_DIR}" \
             --log_dir "${LOG_DIR}" \
             --profiling_dir "${PROFILING_DIR}"
}

# Export the function so srun tasks can see it
export -f run_training

# CPU affinity
CPU_BIND="map_cpu:1,73,145,217"

# --- Execute with Apptainer ---
time srun --cpu-bind=${CPU_BIND} apptainer exec --nv \
      -B "${PROJECT_DIR}:/workspace" \
      -B "${CLUSTER_PROJECT_ROOT}" \
      "${APPTAINER_SIF}" bash -c run_training

echo
echo "--- Finished :) ---"
