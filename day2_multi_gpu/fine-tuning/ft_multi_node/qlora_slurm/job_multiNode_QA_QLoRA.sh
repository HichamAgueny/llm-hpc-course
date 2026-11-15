#!/bin/bash -e
#SBATCH --job-name=ft-llama3-1B-qlora-8gpu
#SBATCH --time=00:10:00
#SBATCH --partition=accel
#SBATCH --nodes=2
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
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

# Load cluster high-performance communication environment components
module load "${NRIS_GPU_MODULE}"
module load "${NCCL_MODULE}"

export PYTHONNOUSERSITE=1

# =====================================================================
# --- LAB SETTINGS (paths, model, dataset and container: see course_env.sh) ---
# =====================================================================
export RUN_NAME="llama3.2-1b_qlora_multi_node"
export CONFIG_FILE="${COURSE_ROOT}/configs/qlora/llama3_2_1B_qlora_multi_node_QA.yaml"
export PYTHON_FILE="${COURSE_ROOT}/recipes/distributed/lora_finetune_distributed.py"

# Outputs go to results/checkpoint_results/<RUN_NAME>/
export OUTPUT_DIR="${CHECKPOINT_RESULTS_DIR}/${RUN_NAME}/checkpoints"
export LOG_DIR="${CHECKPOINT_RESULTS_DIR}/${RUN_NAME}/logs"
export PROFILING_DIR="${CHECKPOINT_RESULTS_DIR}/${RUN_NAME}/profiles"
# =====================================================================

# --- Host network libraries mounted into the container (defaults in course_env.sh;
#     the NCCL module, when loaded, provides the exact paths) ---
export HOST_NCCL_PATH="${NCCL_HOME:-${HOST_NCCL_PATH}}"
if [[ -n "${EBROOTAWSMINOFIMINNCCL:-}" ]]; then
    export HOST_AWS_OFI_NCCL_PATH="${EBROOTAWSMINOFIMINNCCL}/lib"
fi

course_require "${PYTORCH_SIF}" "PyTorch container image" "run ./setup_env.sh or check COURSE_ASSETS_DIR in course_env.sh"
course_require "${BASE_MODEL_DIR}" "Base model" "run ./setup_env.sh or check COURSE_ASSETS_DIR in course_env.sh"
course_require "${QA_DATASET_DIR}" "QA dataset" "run ./setup_env.sh or check COURSE_ASSETS_DIR in course_env.sh"

echo "--- Course folder (host): ${COURSE_ROOT}  (also mounted as /workspace)"
echo

# --- Slurm Settings Diagnostics ---
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "--nbr of nodes: $SLURM_JOB_NUM_NODES"
echo "--nbr of GPUs per node: $SLURM_GPUS_PER_NODE"
echo

# --- Resolve Multi-Node Coordination Addresses ---
nodes=( $(scontrol show hostnames $SLURM_JOB_NODELIST) )
head_node_ip=$(getent hosts "${nodes[0]}" | awk '{print $1}')

export MASTER_ADDR="$head_node_ip"
export MASTER_PORT=25900
export WORLD_SIZE=$SLURM_NTASKS
export LOCAL_WORLD_SIZE=$SLURM_NTASKS_PER_NODE

echo "--WORLD_SIZE: $WORLD_SIZE"
echo "--LOCAL_WORLD_SIZE: $LOCAL_WORLD_SIZE"
echo
echo "=== Running inside Apptainer ==="
echo "CONFIG_FILE: ${CONFIG_FILE}"
echo "PYTHON_FILE: ${PYTHON_FILE}"
echo "CONTAINER:   ${PYTORCH_SIF}"
echo

echo "--- Launching the application inside Apptainer ---"

# --- Define the training task function ---
run_training() {
    export NCCL_DEBUG=INFO
    # Force output streaming directly into your Slurm file
    export PYTHONUNBUFFERED=1

    # These resolve dynamically on each specific task/GPU node during execution
    export RANK=$SLURM_PROCID
    export LOCAL_RANK=$SLURM_LOCALID

    echo "Task ${SLURM_PROCID}: RANK=${RANK}, LOCAL_RANK=${LOCAL_RANK}, WORLD_SIZE=${WORLD_SIZE}, LOCAL_WORLD_SIZE=${LOCAL_WORLD_SIZE}"
    echo "LOCAL_RANK: ${LOCAL_RANK}, CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"

    # Slingshot / NCCL network settings (shared by all multi-node labs)
    source "${COURSE_ROOT}/utils/nccl_env.sh"

    # Link the mapped host communication libraries into the runtime linker path
    export LD_LIBRARY_PATH="/opt/nccl/lib:/opt/aws-ofi-nccl/lib:/opt/libfabric/lib:/opt/cxi:${LD_LIBRARY_PATH}"

    python "${PYTHON_FILE}" --config "${CONFIG_FILE}" \
            --checkpoint_dir "${BASE_MODEL_DIR}" \
            --tokenizer_dir "${TOKENIZER_DIR}" \
            --dataset_dir "${QA_DATASET_DIR}" \
            --output_dir "${OUTPUT_DIR}" \
            --log_dir "${LOG_DIR}" \
            --profiling_dir "${PROFILING_DIR}"
}

# Export the function so srun context forks can target it directly
export -f run_training

# CPU affinity configuration
CPU_BIND="map_cpu:1,73,145,217"

# --- Execute with Apptainer ---
time srun --cpu-bind=${CPU_BIND} --network=disable_rdzv_get apptainer exec --nv ${CONTAINER_BINDS} \
      --bind /cluster/software \
      --bind "${HOST_NCCL_PATH}:/opt/nccl" \
      --bind "${HOST_AWS_OFI_NCCL_PATH}:/opt/aws-ofi-nccl/lib" \
      --bind "${HOST_LIBFABRIC_LIB_PATH}:/opt/libfabric/lib" \
      --bind "${HOST_LIBFABRIC_INCLUDE_PATH}:/opt/libfabric/include" \
      --bind "${HOST_CXI_LIB_PATH}:/opt/cxi" \
      "${PYTORCH_SIF}" bash -c run_training

echo
echo "--- Finished :) ---"
echo "Checkpoints: ${OUTPUT_DIR}"
echo "Logs:        ${LOG_DIR}"
