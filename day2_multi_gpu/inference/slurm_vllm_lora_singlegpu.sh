#!/bin/bash -e
#SBATCH --job-name=vllm-1gpu
#SBATCH --account=nn9971k
#SBATCH --time=00:10:00
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

export NUMEXPR_MAX_THREADS=72


# =====================================================================
# --- CONFIGURABLE VARIABLES AND PATHS (USERS ONLY MODIFY THIS BLOCK) ---
# =====================================================================

# 1. Base Project Allocation
export CLUSTER_PROJECT_ROOT="/cluster/work/projects/nn9971k"

# 2. Personal Workspace Setup
export PROJECT_DIR="${CLUSTER_PROJECT_ROOT}/${USER}/llm-hpc-course"
export SHARED_DIR="${PROJECT_DIR}/shared"
export RESULT_DIR="${PROJECT_DIR}/results/checkpoint_results"
export CURRENT_DIR="${PROJECT_DIR}/day2_multi_gpu/inference"

# 3. Model, Adapter & Serving Parameters
export BASE_MODEL_DIR="${MODEL_PATH:-${SHARED_DIR}/models/Llama-3.2-1B-Instruct}"
export LORA_ADAPTER_DIR="${LORA_PATH:-${RESULT_DIR}/llama3.2-1b_lora_single_device/checkpoints/epoch_1}"
export PROMPT_FILE="${PROMPT_FILE:-"${CURRENT_DIR}/prompt_QA.json"}"
export QUANTIZATION="${QUANTIZATION:-None}"  # Set to "bitsandbytes" for QLoRA
export MAX_LORA_RANK="${MAX_LORA_RANK:-64}"

# 4. Apptainer Container Configuration
export CONTAINER_DIR="${PROJECT_DIR}/apptainer"
export APPTAINER_SIF="${CONTAINER_DIR}/vllm0.12_cu131_py3.12_arm_custom.sif"

# 5. Script & Parallelism Settings
export PYTHON_FILE="${PROJECT_DIR}/recipes/inference/vllm_inference.py"

# 6. Caches under project share
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

# --- Define the inference task function ---
run_inference() {

    python "${PYTHON_FILE}" \
        --model "$BASE_MODEL_DIR" \
        --lora-path "$LORA_ADAPTER_DIR" \
        --prompt-file "$PROMPT_FILE" \
        --quantization "$QUANTIZATION"
}

# Export the function so srun tasks can see it
export -f run_inference

echo "----------------------------------------"
echo "Configuration:"
echo "  Base Model: $BASE_MODEL_DIR"
echo "  LoRA Adapter: $LORA_ADAPTER_DIR"
echo "  Prompt file: $PROMPT_FILE"
echo "  Quantization: $QUANTIZATION"
echo "  Max LoRA Rank: $MAX_LORA_RANK"
echo "----------------------------------------"

# --- Execute with Apptainer ---
time srun apptainer exec --nv \
      -B "${PROJECT_DIR}:/workspace" \
      -B "${CLUSTER_PROJECT_ROOT}" \
      "${APPTAINER_SIF}" bash -c run_inference
echo
echo "--- Finished :) ---"
echo "Job finished at: $(date)"
