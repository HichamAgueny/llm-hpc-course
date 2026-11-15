#!/bin/bash -e
#SBATCH --job-name=vllm-distributed_eessi
#SBATCH --account=nn9997k
#SBATCH --time=00:10:00
#SBATCH --partition=accel
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH -o ./out/%x-%j.out
#SBATCH --mem-per-gpu=96G
#SBATCH --reservation=software

echo "--Node: $(hostname)"
echo

export PYTHONNOUSERSITE=1

ml EESSI/2025.06
#module load torchtune/0.6.1-foss-2025b-PyTorch-2.12.0-CUDA-12.9.1
module load vllm/0.22.1-foss-2025b-PyTorch-2.12.0-CUDA-12.9.1
module load Transformers
module load accelerate

export NUMEXPR_MAX_THREADS=72


# =====================================================================
# --- CONFIGURABLE VARIABLES AND PATHS (USERS ONLY MODIFY THIS BLOCK) ---
# =====================================================================

# 1. Base Project Allocation
export CLUSTER_PROJECT_ROOT="/cluster/projects/nn9997k"

# 2. Personal Workspace Setup
export PROJECT_DIR="${CLUSTER_PROJECT_ROOT}/${USER}/llm-hpc-course"
export SHARED_DIR="${PROJECT_DIR}/shared"
export RESULT_DIR="${PROJECT_DIR}/results/checkpoint_results"
export CURRENT_DIR="${PROJECT_DIR}/day2_multi_gpu/inference"

# 3. Model, Adapter & Serving Parameters
export BASE_MODEL_DIR="${MODEL_PATH:-${SHARED_DIR}/models/Llama-3.2-1B-Instruct}"
export LORA_ADAPTER_DIR="${LORA_PATH:-${RESULT_DIR}/eessi_llama3.2-1b_lora_single_device/checkpoints/epoch_0}"
export PROMPT_FILE="${PROMPT_FILE:-"${CURRENT_DIR}/prompt_QA.json"}"
export QUANTIZATION="${QUANTIZATION:-None}"  # Set to "bitsandbytes" for QLoRA
export MAX_LORA_RANK="${MAX_LORA_RANK:-64}"

# 4. Script & Parallelism Settings
export PYTHON_FILE="${PROJECT_DIR}/recipes/inference/vllm_distributed.py"
export TP_SIZE=4  # Tensor parallel = number of GPUs per node (must match --ntasks-per-node)
export PP_SIZE=1  # Pipeline parallel (use >1 for multi-node)

# 5. Caches under project share
export HF_HOME="${PROJECT_DIR}/hf_cache"
export VLLM_CACHE_ROOT="${PROJECT_DIR}/.cache/vllm"
mkdir -p "${HF_HOME}" "${VLLM_CACHE_ROOT}"

# Logging level
export VLLM_LOGGING_LEVEL=ERROR

# =====================================================================


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

# Verify prompt file exists
if [[ ! -f "$PROMPT_FILE" ]]; then
    echo "ERROR: Prompt file not found: $PROMPT_FILE"
    exit 1
fi

# Set up variables to control distributed PyTorch execution
export MASTER_ADDR=$(hostname)
export MASTER_PORT=25900
export WORLD_SIZE=$SLURM_NPROCS
export LOCAL_WORLD_SIZE=$SLURM_GPUS_PER_NODE

# --- Define the inference task function ---
run_inference() {
    # These resolve dynamically on each task/GPU node during srun
    export RANK=$SLURM_PROCID
    export LOCAL_RANK=$SLURM_LOCALID

    echo "Task ${SLURM_PROCID}: RANK=${RANK}, LOCAL_RANK=${LOCAL_RANK}, WORLD_SIZE=${WORLD_SIZE}, LOCAL_WORLD_SIZE=${LOCAL_WORLD_SIZE}"
    echo "LOCAL_RANK: ${LOCAL_RANK}, CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"

    python "${PYTHON_FILE}" \
        --model "$BASE_MODEL_DIR" \
        --lora-path "$LORA_ADAPTER_DIR" \
        --prompt-file "$PROMPT_FILE" \
        --tensor-parallel-size "$TP_SIZE" \
        --pipeline-parallel-size "$PP_SIZE"
}

# Export the function so srun tasks can see it
export -f run_inference

# CPU affinity mapping
CPU_BIND="map_cpu:1,73,145,217"

echo "----------------------------------------"
echo "Configuration:"
echo "  Base Model: $BASE_MODEL_DIR"
echo "  LoRA Adapter: $LORA_ADAPTER_DIR"
echo "  Prompt file: $PROMPT_FILE"
echo "  Quantization: $QUANTIZATION"
echo "  Max LoRA Rank: $MAX_LORA_RANK"
echo "----------------------------------------"

# --- Execute with srun ---
time srun --cpu-bind=${CPU_BIND} bash -c run_inference

echo
echo "--- Finished :) ---"
echo "Job finished at: $(date)"
