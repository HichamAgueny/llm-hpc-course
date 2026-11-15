#!/bin/bash -e
#SBATCH --job-name=vllm-1gpu
#SBATCH --time=00:10:00
#SBATCH --partition=accel
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH -o ./out/%x-%j.out
#SBATCH --mem-per-gpu=96G
# Slurm account & reservation are set in course_env.sh - no need to add them here.

# --- Load the course settings (account, paths, containers, modules) ----------
for _d in "${SLURM_SUBMIT_DIR:-$PWD}" "$PWD" "${COURSE_ROOT:-/}"; do
    until [[ -f "$_d/course_env.sh" || "$_d" == "/" ]]; do _d="$(dirname "$_d")"; done
    if [[ -f "$_d/course_env.sh" ]]; then break; fi
done
if [[ ! -f "$_d/course_env.sh" ]]; then echo "ERROR: course_env.sh not found - submit from inside the course folder" >&2; exit 1; fi
source "$_d/course_env.sh"

echo "--Node: $(hostname)"
echo

export PYTHONNOUSERSITE=1

export NUMEXPR_MAX_THREADS=72

# =====================================================================
# --- LAB SETTINGS (paths, model, adapter and container: see course_env.sh) ---
# =====================================================================
# Model and adapter: override at submit time with MODEL_PATH=... / LORA_PATH=...
export BASE_MODEL_DIR="${MODEL_PATH:-${BASE_MODEL_DIR}}"
export PROMPT_FILE="${PROMPT_FILE:-${COURSE_ROOT}/day2_multi_gpu/inference/prompt_QA.json}"
export QUANTIZATION="${QUANTIZATION:-None}"  # Set to "bitsandbytes" for QLoRA
export MAX_LORA_RANK="${MAX_LORA_RANK:-64}"

export PYTHON_FILE="${COURSE_ROOT}/recipes/inference/vllm_inference.py"

mkdir -p "${HF_HOME}" "${VLLM_CACHE_ROOT}"

# Logging level
export VLLM_LOGGING_LEVEL=ERROR
# =====================================================================

course_require "${VLLM_SIF}" "vLLM container image" "run ./setup_env.sh or check COURSE_ASSETS_DIR in course_env.sh"
course_require "${BASE_MODEL_DIR}" "Base model" "run ./setup_env.sh or check COURSE_ASSETS_DIR in course_env.sh"
course_require "${LORA_ADAPTER_DIR}" "LoRA adapter" "fine-tune first (./run.sh day1 lora) or choose one with LORA_PATH=/path/to/epoch_N"
course_require "${PROMPT_FILE}" "Prompt file"

# --- Slurm setting
N=$SLURM_JOB_NUM_NODES
total_nbr_gpus=$(($N * $SLURM_GPUS_ON_NODE))
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "--nbr of nodes: $N"
echo "--nbr of GPUs per node: $SLURM_GPUS_ON_NODE"
echo "--Total nbr of GPUs: $total_nbr_gpus"
echo

# --- Locale Settings ---
export LANG=en_US.UTF-8
export LC_ALL=en_US.UTF-8

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
time srun apptainer exec --nv ${CONTAINER_BINDS} \
      "${VLLM_SIF}" bash -c run_inference
echo
echo "--- Finished :) ---"
echo "Job finished at: $(date)"
