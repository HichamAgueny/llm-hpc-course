#!/bin/bash -e
#SBATCH --job-name=mmlu_eval-llama3.2-1B-base-1gpu_epoch0
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

export NUMEXPR_MAX_THREADS=72

# =====================================================================
# --- LAB SETTINGS (paths, model, adapter and container: see course_env.sh) ---
# =====================================================================
# Base model evaluation: no LoRA, no merge.
# The eval YAML reads the model from MERGED_MODEL_DIR, so point it at the base model.
export MERGED_MODEL_DIR="${BASE_MODEL_DIR}"
export OUTPUT_DIR="${EVAL_RESULTS_DIR}/llama3.2-1b-base_mmlu_singlegpu"
export CONFIG_FILE="${COURSE_ROOT}/configs/eval_cfg/eval_config.yaml"
export PYTHON_FILE_EVAL="${COURSE_ROOT}/recipes/eval/eleuther_eval.py"

mkdir -p "${OUTPUT_DIR}" "${HF_HOME}"
# =====================================================================

course_require "${PYTORCH_SIF}" "PyTorch container image" "run ./setup_env.sh or check COURSE_ASSETS_DIR in course_env.sh"
course_require "${BASE_MODEL_DIR}" "Base model" "run ./setup_env.sh or check COURSE_ASSETS_DIR in course_env.sh"

echo "--- Course folder: ${COURSE_ROOT}"
echo
echo "=== Evaluation configuration ==="
echo "MODEL EVALUATED: ${MERGED_MODEL_DIR}"
echo "CONFIG_FILE:     ${CONFIG_FILE}"
echo "RESULTS:         ${OUTPUT_DIR}"
echo


# =====================================================================
# Evaluation Sequence
# =====================================================================
echo "--- Launching Evaluation Harness ---"

time srun apptainer exec --nv ${CONTAINER_BINDS} \
      "${PYTORCH_SIF}" \
      python "${PYTHON_FILE_EVAL}" \
      --config "${CONFIG_FILE}" \
      tasks=["mmlu_philosophy","mmlu_sociology","mmlu_business_ethics","mmlu_college_physics"] \
      batch_size=8 \
      save_results=True

echo
echo "--- Finished :) ---"
echo "Results: ${OUTPUT_DIR}"
# tasks=["mmlu"]
# tasks='["hellaswag", "truthfulqa_mc2", "gsm8k"]'
