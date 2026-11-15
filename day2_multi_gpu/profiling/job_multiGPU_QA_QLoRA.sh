#!/bin/bash -e
#SBATCH --job-name=ft-llama3-1B-qlora-4gpu_profiling
#SBATCH --time=00:15:00
#SBATCH --partition=accel
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH -o ./out/%x-%j.out
#SBATCH --mem-per-gpu=97G
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

# =====================================================================
# --- LAB SETTINGS (paths, model, dataset and container: see course_env.sh) ---
# =====================================================================
export RUN_NAME="profiling_llama3.2-1b_qlora_multi_device"
export CONFIG_FILE="${COURSE_ROOT}/configs/qlora/llama3_2_1B_qlora_multi_device_profiling.yaml"
export PYTHON_FILE="${COURSE_ROOT}/recipes/distributed/lora_finetune_distributed.py"

# Profiles and logs go to results/profiling_results/<RUN_NAME>/
# (checkpoints are throw-away here, so they go to the node-local /tmp)
export OUTPUT_DIR="/tmp/${USER}_${SLURM_JOB_ID}/checkpoints"
export LOG_DIR="${PROFILING_RESULTS_DIR}/${RUN_NAME}/logs"
export PROFILING_DIR="${PROFILING_RESULTS_DIR}/${RUN_NAME}/profiles"
# =====================================================================

course_require "${PYTORCH_SIF}" "PyTorch container image" "run ./setup_env.sh or check COURSE_ASSETS_DIR in course_env.sh"
course_require "${BASE_MODEL_DIR}" "Base model" "run ./setup_env.sh or check COURSE_ASSETS_DIR in course_env.sh"
course_require "${QA_DATASET_DIR}" "QA dataset" "run ./setup_env.sh or check COURSE_ASSETS_DIR in course_env.sh"

echo "--- Course folder (host): ${COURSE_ROOT}  (also mounted as /workspace)"
echo
echo "=== Running inside Apptainer ==="
echo "CONFIG_FILE: ${CONFIG_FILE}"
echo "PYTHON_FILE: ${PYTHON_FILE}"
echo "CONTAINER:   ${PYTORCH_SIF}"
echo

# Flash Attention for efficiency
export USE_FLASH_ATTENTION=1
# Suppress LMOD debugging, set locale
export LMOD_SH_DBG_ON=0
export LANG=en_US.UTF-8
export LC_ALL=en_US.UTF-8

echo "--- Launching the application inside Apptainer ---"

# Set up variables to control distributed PyTorch training
export MASTER_ADDR=$(hostname)
export MASTER_PORT=25900
export WORLD_SIZE=$SLURM_NPROCS
export LOCAL_WORLD_SIZE=$SLURM_GPUS_PER_NODE

# --- Define the training task function ---
run_training() {

    # These resolve dynamically on each task/GPU node during srun
    export RANK=$SLURM_PROCID
    export LOCAL_RANK=$SLURM_LOCALID

    echo "Task ${SLURM_PROCID}: RANK=${RANK}, LOCAL_RANK=${LOCAL_RANK}, WORLD_SIZE=${WORLD_SIZE}, LOCAL_WORLD_SIZE=${LOCAL_WORLD_SIZE}"
    echo "LOCAL_RANK: ${LOCAL_RANK}, CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"

    python "${PYTHON_FILE}" --config "${CONFIG_FILE}" \
             --checkpoint_dir "${BASE_MODEL_DIR}" \
             --tokenizer_dir "${TOKENIZER_DIR}" \
             --dataset_dir "${QA_DATASET_DIR}" \
             --output_dir "${OUTPUT_DIR}" \
             --log_dir "${LOG_DIR}" \
             --profiling_dir "${PROFILING_DIR}"
}

# Export the function so srun tasks can see it
export -f run_training

# CPU affinity
CPU_BIND="map_cpu:1,73,145,217"

# --- Execute with Apptainer ---
time srun --cpu-bind=${CPU_BIND} apptainer exec --nv ${CONTAINER_BINDS} \
      "${PYTORCH_SIF}" bash -c run_training

echo
echo "--- Finished :) ---"
echo "Profiles: ${PROFILING_DIR}"
echo "Logs:     ${LOG_DIR}"
