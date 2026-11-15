#!/bin/bash -e
#SBATCH --job-name=ft-llama3.2-1B-lora-1gpu_eessi
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

export NUMEXPR_MAX_THREADS=72

# =====================================================================
# --- LAB SETTINGS (paths, model and dataset: see course_env.sh) ---
# =====================================================================
export RUN_NAME="eessi_llama3.2-1b_qlora_single_device"
export CONFIG_FILE="${COURSE_ROOT}/configs/qlora/llama3_2_1B_qlora_single_device_QA.yaml"
export PYTHON_FILE="${COURSE_ROOT}/recipes/single_device/lora_finetune_single_device.py"

# Outputs go to results/checkpoint_results/<RUN_NAME>/
export OUTPUT_DIR="${CHECKPOINT_RESULTS_DIR}/${RUN_NAME}/checkpoints"
export LOG_DIR="${CHECKPOINT_RESULTS_DIR}/${RUN_NAME}/logs"
export PROFILING_DIR="${CHECKPOINT_RESULTS_DIR}/${RUN_NAME}/profiles"
# =====================================================================

course_require "${BASE_MODEL_DIR}" "Base model" "run ./setup_env.sh or check COURSE_ASSETS_DIR in course_env.sh"
course_require "${QA_DATASET_DIR}" "QA dataset" "run ./setup_env.sh or check COURSE_ASSETS_DIR in course_env.sh"

echo "--- Course folder: ${COURSE_ROOT}"
echo

echo "=== Configuration ==="
echo "CONFIG_FILE: ${CONFIG_FILE}"
echo "PYTHON_FILE: ${PYTHON_FILE}"
echo

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

# --- Execute with srun ---
time srun python "${PYTHON_FILE}" --config "${CONFIG_FILE}" \
    --checkpoint_dir "${BASE_MODEL_DIR}" \
    --tokenizer_dir "${TOKENIZER_DIR}" \
    --dataset_dir "${QA_DATASET_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --log_dir "${LOG_DIR}" \
    --profiling_dir "${PROFILING_DIR}"

echo
echo "--- Finished :) ---"
echo "Checkpoints: ${OUTPUT_DIR}"
echo "Logs:        ${LOG_DIR}"
