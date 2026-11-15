#!/bin/bash -e
#SBATCH --job-name=qat-ao-llama3-1B-lora-1gpu
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
module load torchao

export NUMEXPR_MAX_THREADS=72

# =====================================================================
# --- LAB SETTINGS (paths and model: see course_env.sh) ---
# =====================================================================
# Input: the shared base model
export MODEL_DIR="${BASE_MODEL_DIR}"

# Output: your own copy under results/quantized_models/
# (the serving labs pick it up from there automatically)
export OUTPUT_DIR="${QUANTIZED_MODELS_DIR}/${BASE_MODEL_NAME}-torchao_int8"
export OUT_DIR="${OUTPUT_DIR}"

export PYTHON_FILE="${COURSE_ROOT}/recipes/quanti/qat_torchoa.py"

mkdir -p "${OUTPUT_DIR}" "${HF_HOME}"
# =====================================================================

course_require "${BASE_MODEL_DIR}" "Base model" "run ./setup_env.sh or check COURSE_ASSETS_DIR in course_env.sh"

echo "--- Course folder: ${COURSE_ROOT}"
echo

echo "=== Configuration ==="
echo "PYTHON_FILE: ${PYTHON_FILE}"
echo "MODEL_DIR: ${MODEL_DIR}"
echo "OUTPUT_DIR: ${OUTPUT_DIR}"
echo

# --- Slurm setting
N=$SLURM_JOB_NUM_NODES
total_nbr_gpus=$(($N * ${SLURM_GPUS_ON_NODE:-1}))
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "--nbr of nodes: $N"
echo "--nbr of GPUs per node: ${SLURM_GPUS_ON_NODE:-1}"
echo "--Total nbr of GPUs: $total_nbr_gpus"
echo

# --- Locale Settings ---
export LANG=en_US.UTF-8
export LC_ALL=en_US.UTF-8

# --- Execute with srun ---
# INT8 weight-only quantization via torchao:
# checkpoint (bf16)  ->  loaded model (bf16)  ->  quantized (int8 weights)
# https://docs.pytorch.org/ao/stable/api_reference/generated/torchao.quantization.Int8WeightOnlyConfig.html
time srun python "${PYTHON_FILE}" --model_path "${MODEL_DIR}" --output_path "${OUTPUT_DIR}"

echo
echo "--- Finished :) ---"
echo "Quantized model: ${OUTPUT_DIR}"
echo "Compare sizes:   du -sh ${BASE_MODEL_DIR} ${OUTPUT_DIR}"
