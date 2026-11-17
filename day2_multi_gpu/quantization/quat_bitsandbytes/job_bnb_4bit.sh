#!/bin/bash -e
#SBATCH --job-name=qat-llama3-1B-lora-1gpu_4bits
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
#SBATCH --reservation=LLM_day2

echo "--Node: $(hostname)"
echo

export PYTHONNOUSERSITE=1

ml EESSI/2025.06
module load torchtune/0.6.1-foss-2025b-PyTorch-2.12.0-CUDA-12.9.1
module load Transformers
module load accelerate
module load bitsandbytes

export NUMEXPR_MAX_THREADS=72

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
export MODEL_DIR="${SHARED_DIR}/models/Llama-3.2-1B-Instruct"

# 4. Output Component Paths
export OUTPUT_DIR="${SHARED_DIR}/models/Llama-3.2-1B-Instruct-bnb_4bit"
export OUT_DIR="${OUTPUT_DIR}"

# 5. Script / Config Layout
export PYTHON_FILE="${PROJECT_DIR}/recipes/quanti/quat_bnb_4bit.py"

# 6. HuggingFace caches (datasets + hub) under the project share
export HF_HOME="${PROJECT_DIR}/hf_cache"
mkdir -p "${HF_HOME}"
# =====================================================================

# --- Other Quantization Options ---
DTYPE="auto"                                        # "auto", "float16", or "bfloat16"
DEVICE="auto"                                       # Strategy to spread weights over available GPUs

# Check if output directory exists; if not, create it
if [ ! -d "${OUTPUT_DIR}" ]; then
    echo "Output directory does not exist. Creating: ${OUTPUT_DIR}"
    mkdir -p "${OUTPUT_DIR}"
fi

echo "--- My Main Directory (host): ${PROJECT_DIR}"
echo
echo "=== Configuration ==="
echo "PYTHON_FILE: ${PYTHON_FILE}"
echo "MODEL_DIR: ${MODEL_DIR}"
echo "OUTPUT_DIR: ${OUTPUT_DIR}"
echo

# --- Slurm setting
N=$SLURM_JOB_NUM_NODES
nproc_perN=$SLURM_NTASKS_PER_NODE
total_nbr_gpus=$(($N * ${SLURM_GPUS_ON_NODE:-1}))
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "--nbr of nodes: $N"
echo "--nbr of GPUs per node: ${SLURM_GPUS_ON_NODE:-1}"
echo "--Total nbr of GPUs: $total_nbr_gpus"
echo

echo "--SLURM_PROCID: $SLURM_PROCID"
echo

# --- Locale Settings ---
export LANG=en_US.UTF-8
export LC_ALL=en_US.UTF-8

# --- Execute with srun ---
time srun python "${PYTHON_FILE}" \
    --model_path "${MODEL_DIR}" \
    --output_path "${OUTPUT_DIR}" \
    --dtype "$DTYPE" \
    --device_map "$DEVICE"

echo
echo "--- Finished :) ---"
