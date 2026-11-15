#!/bin/bash -e
#SBATCH --job-name=mmlu_eval-llama3.2-1B-lora-1gpu_epoch0_eessi
#SBATCH --time=00:30:00
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

module load "${EESSI_MODULE}"
module load "${TORCHTUNE_MODULE}"
module load Transformers
module load accelerate
module load "${LM_EVAL_MODULE}"

export NUMEXPR_MAX_THREADS=72

# =====================================================================
# --- LAB SETTINGS (paths, model, adapter and container: see course_env.sh) ---
# =====================================================================
# Evaluates LORA_ADAPTER_DIR (default: the Day 1 LoRA adapter). Pick another
# adapter at submit time with:  LORA_PATH=/path/to/checkpoints/epoch_N ./run.sh ...
# The merged model is written to MERGED_MODEL_DIR (= <adapter>_merged).
export OUTPUT_DIR="${EVAL_RESULTS_DIR}/llama3.2-1b-mmlu_singlegpu_eessi"
export CONFIG_FILE="${COURSE_ROOT}/configs/eval_cfg/eval_config.yaml"
export PYTHON_FILE_EVAL="${COURSE_ROOT}/recipes/eval/eleuther_eval.py"
export PYTHON_FILE_MERGE="${COURSE_ROOT}/recipes/merge/merge_and_save.py"
export MAX_SHARD_SIZE="5GB"
mkdir -p "${OUTPUT_DIR}" "${HF_HOME}"
# =====================================================================

course_require "${BASE_MODEL_DIR}" "Base model" "run ./setup_env.sh or check COURSE_ASSETS_DIR in course_env.sh"

echo "--- Course folder: ${COURSE_ROOT}"
echo
echo "=== Evaluation configuration ==="
echo "MODEL EVALUATED: ${MERGED_MODEL_DIR}"
echo "LORA ADAPTER:    ${LORA_ADAPTER_DIR}"
echo "CONFIG_FILE:     ${CONFIG_FILE}"
echo "RESULTS:         ${OUTPUT_DIR}"
echo

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

    time srun python "${PYTHON_FILE_MERGE}" \
                --base_model_path "${BASE_MODEL_DIR}" \
                --lora_adapter_path "${LORA_ADAPTER_DIR}" \
                --output_path "${MERGED_MODEL_DIR}" \
                --max_shard_size "${MAX_SHARD_SIZE}"

    echo "--Weight merge processing complete :)"
fi

echo

# =====================================================================
# STEP 2: Evaluation Sequence
# =====================================================================
echo "--- Launching Evaluation Harness ---"

time srun python "${PYTHON_FILE_EVAL}" \
      --config "${CONFIG_FILE}" \
      tasks=["mmlu_philosophy","mmlu_sociology","mmlu_business_ethics","mmlu_college_physics"] \
      batch_size=8 \
      save_results=True

echo
echo "--- Finished :) ---"
echo "Results: ${OUTPUT_DIR}"
# tasks=["mmlu"]
# tasks='["hellaswag", "truthfulqa_mc2", "gsm8k"]'
