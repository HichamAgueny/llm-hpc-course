#!/bin/bash -e
#SBATCH --job-name=eval-llama3.2-1B-lora-4gpu_mmlu
#SBATCH --time=00:30:00
#SBATCH --partition=accel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --mem-per-gpu=96G
#SBATCH -o ./out/%x-%j.out
# Slurm account & reservation are set in course_env.sh - no need to add them here.

# --- Load the course settings (account, paths, containers, modules) ----------
for _d in "${SLURM_SUBMIT_DIR:-$PWD}" "$PWD" "${COURSE_ROOT:-/}"; do
    until [[ -f "$_d/course_env.sh" || "$_d" == "/" ]]; do _d="$(dirname "$_d")"; done
    if [[ -f "$_d/course_env.sh" ]]; then break; fi
done
if [[ ! -f "$_d/course_env.sh" ]]; then echo "ERROR: course_env.sh not found - submit from inside the course folder" >&2; exit 1; fi
source "$_d/course_env.sh"

# ---------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------

export PYTHONNOUSERSITE=1

export NUMEXPR_MAX_THREADS=72

# =====================================================================
# --- LAB SETTINGS (paths, model, adapter and container: see course_env.sh) ---
# =====================================================================
# Evaluates LORA_ADAPTER_DIR (default: the Day 1 LoRA adapter). Pick another
# adapter at submit time with:  LORA_PATH=/path/to/checkpoints/epoch_N ./run.sh ...
# The merged model is written to MERGED_MODEL_DIR (= <adapter>_merged).
export OUTPUT_DIR="${EVAL_RESULTS_DIR}/llama3.2-1b-mmlu_multigpu"
export PYTHON_FILE_MERGE="${COURSE_ROOT}/recipes/merge/merge_and_save.py"
export MAX_SHARD_SIZE="5GB"
mkdir -p "${OUTPUT_DIR}" "${HF_HOME}"
# =====================================================================

course_require "${PYTORCH_SIF}" "PyTorch container image" "run ./setup_env.sh or check COURSE_ASSETS_DIR in course_env.sh"
course_require "${BASE_MODEL_DIR}" "Base model" "run ./setup_env.sh or check COURSE_ASSETS_DIR in course_env.sh"

# ---------------------------------------------------------------------
# GPU information
# ---------------------------------------------------------------------

echo "======================================================"
echo "Starting 4-GPU evaluation"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Allocated GPUs: ${SLURM_GPUS}"
echo "Model evaluated: ${MERGED_MODEL_DIR}"
echo "LoRA adapter: ${LORA_ADAPTER_DIR}"
echo "Results: ${OUTPUT_DIR}"
echo "======================================================"

# =====================================================================
# STEP 1: Merge the LoRA adapter into the base model (skipped if done before)
# =====================================================================
echo "--- Checking for existing merged weights ---"

if [ -f "${MERGED_MODEL_DIR}/config.json" ]; then
    echo "--Found valid merged model at: ${MERGED_MODEL_DIR}"
    echo ">> Skipping merge step to save cluster compute time."
else
    course_require "${LORA_ADAPTER_DIR}" "LoRA adapter" "fine-tune first (./run.sh day1 lora) or choose one with LORA_PATH=/path/to/epoch_N"
    echo "--No merged model found at destination. Initiating merge..."
    echo "--- Launching the merge application inside Apptainer ---"

    time srun apptainer exec --nv ${CONTAINER_BINDS} \
          "${PYTORCH_SIF}" \
          python "${PYTHON_FILE_MERGE}" \
                --base_model_path "${BASE_MODEL_DIR}" \
                --lora_adapter_path "${LORA_ADAPTER_DIR}" \
                --output_path "${MERGED_MODEL_DIR}" \
                --max_shard_size "${MAX_SHARD_SIZE}"

    echo "--Weight merge processing complete :)"
fi

echo

# ---------------------------------------------------------------------
# STEP 2: Run lm-evaluation-harness with Accelerate
# ---------------------------------------------------------------------
# CPU affinity
CPU_BIND="map_cpu:1,73,145,217"

time srun --cpu-bind=${CPU_BIND} \
    --ntasks=1 \
    --gpus=4 \
    --cpus-per-task=72 \
    apptainer exec --nv ${CONTAINER_BINDS} \
        "${PYTORCH_SIF}" \
        bash -lc '
            echo "Starting Accelerate evaluation..."
            echo "Output directory: '"${OUTPUT_DIR}"'"
            accelerate launch \
                --multi_gpu \
                --num_processes 4 \
                -m lm_eval \
                --model hf \
                --model_args "pretrained='"${MERGED_MODEL_DIR}"',dtype=bfloat16" \
                --tasks mmlu_philosophy,mmlu_sociology,mmlu_business_ethics,mmlu_college_physics \
                --apply_chat_template \
                --batch_size 8 \
                --output_path "'"${OUTPUT_DIR}"'/results.json"
        '

echo "======================================================"
echo "Evaluation completed."
echo "Results: ${OUTPUT_DIR}"
echo "======================================================"
