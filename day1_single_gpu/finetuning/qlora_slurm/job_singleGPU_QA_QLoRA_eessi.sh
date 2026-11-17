#!/bin/bash -e
#SBATCH --job-name=ft-llama3.2-1B-lora-1gpu_eessi
#SBATCH --account=nn9971k
#SBATCH --time=00:30:00
#SBATCH --partition=accel
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH -o ./out/%x-%j.out
#SBATCH --mem-per-gpu=96G
#SBATCH --reservation=nn9971k_213

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

# 1. Input Component Paths
CHECKPOINT_DIR="${SHARED_DIR}/models/Llama-3.2-1B-Instruct"
TOKENIZER_DIR="${CHECKPOINT_DIR}/original/tokenizer.model"
DATASET_DIR="${SHARED_DIR}/datasets/GPT-4-LLM/alpaca_gpt4_data.json"

# 2. Output Component Paths (Explicitly adjust folders here as needed)
RUN_NAME="eessi_llama3.2-1b_qlora_single_device"
OUTPUT_DIR="${RESULT_DIR}/${RUN_NAME}/checkpoints"
LOG_DIR="${RESULT_DIR}/${RUN_NAME}/logs"
PROFILING_DIR="${RESULT_DIR}/${RUN_NAME}/profiles"

# 3. Script / Config Layout
export CONFIG_FILE="${PROJECT_DIR}/configs/qlora/llama3_2_1B_qlora_single_device_QA.yaml"
export PYTHON_FILE="${PROJECT_DIR}/recipes/single_device/lora_finetune_single_device.py"

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

# --- Execute with srun ---
# This block maps your top-level choices perfectly to the Python layer.
time srun python "${PYTHON_FILE}" --config "${CONFIG_FILE}" \
  	--checkpoint_dir "${CHECKPOINT_DIR}" \
  	--tokenizer_dir "${TOKENIZER_DIR}" \
  	--dataset_dir "${DATASET_DIR}" \
  	--output_dir "${OUTPUT_DIR}" \
  	--log_dir "${LOG_DIR}" \
  	--profiling_dir "${PROFILING_DIR}"

echo
echo "--- Finished :) ---"
