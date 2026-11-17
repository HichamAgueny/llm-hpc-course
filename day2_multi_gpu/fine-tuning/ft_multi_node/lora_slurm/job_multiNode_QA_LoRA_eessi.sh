#!/bin/bash -e
#SBATCH --job-name=ft-llama3.2-1B-lora-8gpu
#SBATCH --account=nn9971k
#SBATCH --time=00:30:00
#SBATCH --partition=accel
#SBATCH --nodes=2
#SBATCH --gpus=8
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH -o ./out/%x-%j.out
#SBATCH --mem-per-gpu=96G
#SBATCH --reservation=LLM_day2

echo "--Node: $(hostname)"
echo

export PYTHONNOUSERSITE=1

ml EESSI/2025.06
module load torchtune/0.6.1-foss-2025b-PyTorch-2.12.0-CUDA-12.9.1

export NUMEXPR_MAX_THREADS=72


# =====================================================================
# --- CONFIGURABLE VARIABLES AND PATHS (USERS ONLY MODIFY THIS BLOCK) ---
# =====================================================================

# Root directory of the project
PROJECT_DIR="/cluster/work/projects/nn9971k/$USER/llm-hpc-course"
SHARED_DIR="${PROJECT_DIR}/shared"
RESULT_DIR="${PROJECT_DIR}/results/checkpoint_results"

# 1. Input Component Paths (CRITICAL: Added 'export' so srun can see them)
export CHECKPOINT_DIR="${SHARED_DIR}/models/Llama-3.2-1B-Instruct"
export TOKENIZER_DIR="${CHECKPOINT_DIR}/original/tokenizer.model"
export DATASET_DIR="${SHARED_DIR}/datasets/GPT-4-LLM/alpaca_gpt4_data.json"

# 2. Output Component Paths
export RUN_NAME="eessi_llama3.2-1b_lora_multi_node"
export OUTPUT_DIR="${RESULT_DIR}/${RUN_NAME}/checkpoints"
export LOG_DIR="${RESULT_DIR}/${RUN_NAME}/logs"
export PROFILING_DIR="${RESULT_DIR}/${RUN_NAME}/profiles"

# 3. Script / Config Layout
export CONFIG_FILE="${PROJECT_DIR}/configs/lora/llama3_2_1B_lora_multi_node_QA.yaml"
export PYTHON_FILE="${PROJECT_DIR}/recipes/distributed/lora_finetune_distributed.py"

# 7. Internal Multi-Node Network Helper Path
export MyCurrentWD="${PROJECT_DIR}/day2_multi_gpu/fine-tuning/ft_multi_node/lora_slurm"
# =====================================================================


echo "--- My Main Directory (host): ${PROJECT_DIR}"
echo

echo "=== Configuration ==="
echo "CONFIG_FILE: ${CONFIG_FILE}"
echo "PYTHON_FILE: ${PYTHON_FILE}"
echo

# --- Slurm setting
N=$SLURM_JOB_NUM_NODES
nproc_perN=$SLURM_NTASKS_PER_NODE
total_nbr_gpus=$(($N * $SLURM_GPUS_ON_NODE))
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "--nbr of nodes: $N"
echo "--nbr of GPUs per node: $SLURM_GPUS_ON_NODE"
echo "--Total nbr of GPUs: $total_nbr_gpus"
echo

# --- Locale Settings ---
export LANG=en_US.UTF-8
export LC_ALL=en_US.UTF-8

# Set up variables to control distributed PyTorch training
export MASTER_ADDR=$(hostname)
export MASTER_PORT=25900
export WORLD_SIZE=$SLURM_NPROCS
export LOCAL_WORLD_SIZE=$SLURM_GPUS_PER_NODE

# --- Define the training task function ---
run_training() {

    export NCCL_DEBUG=INFO	
    export PYTHONUNBUFFERED=1

    # These resolve dynamically on each task/GPU node during srun
    export RANK=$SLURM_PROCID
    export LOCAL_RANK=$SLURM_LOCALID

    echo "Task ${SLURM_PROCID}: RANK=${RANK}, LOCAL_RANK=${LOCAL_RANK}, WORLD_SIZE=${WORLD_SIZE}, LOCAL_WORLD_SIZE=${LOCAL_WORLD_SIZE}"
    echo "LOCAL_RANK: ${LOCAL_RANK}, CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"

    # Source local cluster network environments inside the shell context
    if [ -f "${MyCurrentWD}/nccl_env.sh" ]; then
        source "${MyCurrentWD}/nccl_env.sh"
    else
        echo "ERROR: ${MyCurrentWD}/nccl_env.sh not found!"
        exit 1
    fi
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

# --- Execute with srun ---
time srun --cpu-bind=${CPU_BIND} --network=disable_rdzv_get bash -c run_training

echo
echo "--- Finished :) ---"
