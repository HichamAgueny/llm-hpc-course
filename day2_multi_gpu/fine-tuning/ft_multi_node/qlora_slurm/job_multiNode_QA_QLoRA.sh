#!/bin/bash -e
#SBATCH --job-name=ft-llama3-1B-qlora-8gpu
##SBATCH --reservation=LLM_in_person_course
#SBATCH --account=nn9997k
#SBATCH --time=00:10:00
#SBATCH --partition=accel
#SBATCH --nodes=2
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH -o ./out/%x-%j.out
#SBATCH --mem-per-gpu=96G
#SBATCH --reservation=software

echo "--Node: $(hostname)"
echo

# Load cluster high-performance communication environment components
module load NRIS/GPU
module load NCCL/2.30.4-GCCcore-14.3.0-CUDA-13.0.0

export PYTHONNOUSERSITE=1

# =====================================================================
# --- CONFIGURABLE VARIABLES AND PATHS (USERS ONLY MODIFY THIS BLOCK) ---
# =====================================================================

# 1. Base Project Allocation (Change this if migrating to a different project share)
export CLUSTER_PROJECT_ROOT="/cluster/projects/nn9997k"

# 2. Personal Workspace Setup (Derived automatically from your root above)
export PROJECT_DIR="${CLUSTER_PROJECT_ROOT}/${USER}/llm-hpc-course"
export SHARED_DIR="${PROJECT_DIR}/shared"
export RESULT_DIR="${PROJECT_DIR}/results/checkpoint_results"

# 3. Input Component Paths
export CHECKPOINT_DIR="${SHARED_DIR}/models/Llama-3.2-1B-Instruct"
export TOKENIZER_DIR="${CHECKPOINT_DIR}/original/tokenizer.model"
export DATASET_DIR="${SHARED_DIR}/datasets/GPT-4-LLM/alpaca_gpt4_data.json"

# 4. Output Component Paths (Explicitly adjust folders here as needed)
export RUN_NAME="llama3.2-1b_qlora_multi_node"
export OUTPUT_DIR="${RESULT_DIR}/${RUN_NAME}/checkpoints"
export LOG_DIR="${RESULT_DIR}/${RUN_NAME}/logs"
export PROFILING_DIR="${RESULT_DIR}/${RUN_NAME}/profiles"

# 5. Script / Config Layout
export CONFIG_FILE="${PROJECT_DIR}/configs/qlora/llama3_2_1B_qlora_multi_node_QA.yaml"
export PYTHON_FILE="${PROJECT_DIR}/recipes/distributed/lora_finetune_distributed.py"

# 6. Apptainer Container Configuration
export CONTAINER_DIR="${PROJECT_DIR}/apptainer"
export APPTAINER_SIF="${CONTAINER_DIR}/pytorch_25.08_cuda13.0_arm_custom.sif"

# 7. Internal Multi-Node Network Helper Path
export MyCurrentWD="${PROJECT_DIR}/day2_multi_gpu/fine-tuning/ft_multi_node/qlora_slurm"

# =====================================================================

# --- System & High-Speed Network Interconnect Paths (Bypassed by Users) ---
export HOST_NCCL_PATH="${NCCL_HOME:-/cluster/software/NRIS/neoverse_v2/software/NCCL/2.30.4-GCCcore-14.3.0-CUDA-13.0.0}"
export HOST_AWS_OFI_NCCL_PATH="${EBROOTAWSMINOFIMINNCCL:-/cluster/software/NRIS/neoverse_v2/software/aws-ofi-nccl/1.19.1-GCCcore-14.3.0-CUDA-13.0.0}/lib"
export HOST_LIBFABRIC_LIB_PATH="/opt/cray/libfabric/2.3.1/lib64"
export HOST_LIBFABRIC_INCLUDE_PATH="/opt/cray/libfabric/2.3.1/include"
export HOST_CXI_LIB_PATH="/usr/lib64"

echo "--- My Main Directory (host): ${PROJECT_DIR}"
echo "--- Bind-mounted inside container as: /workspace"
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

    # Source local cluster network environments inside the shell context
    if [ -f "${MyCurrentWD}/nccl_env.sh" ]; then
        source "${MyCurrentWD}/nccl_env.sh"
    else
        echo "ERROR: ${MyCurrentWD}/nccl_env.sh not found!"
        exit 1
    fi

    # Link the mapped host communication drives directly into the runtime library linker path
    export LD_LIBRARY_PATH="/opt/nccl/lib:/opt/aws-ofi-nccl/lib:/opt/libfabric/lib:/opt/cxi:${LD_LIBRARY_PATH}"

    # Execute custom python parameter structure matching the single node setup
    python "${PYTHON_FILE}" --config "${CONFIG_FILE}" \
            --checkpoint_dir "${CHECKPOINT_DIR}" \
            --tokenizer_dir "${TOKENIZER_DIR}" \
            --dataset_dir "${DATASET_DIR}" \
            --output_dir "${OUTPUT_DIR}" \
            --log_dir "${LOG_DIR}" \
            --profiling_dir "${PROFILING_DIR}"
}

# Export the function so srun context forks can target it directly
export -f run_training

# CPU affinity configuration 
CPU_BIND="map_cpu:1,73,145,217"

# --- Execute with Apptainer ---
time srun --cpu-bind=${CPU_BIND} --network=disable_rdzv_get apptainer exec --nv \
      --bind "${PROJECT_DIR}:/workspace" \
      --bind "${CLUSTER_PROJECT_ROOT}" \
      --bind /cluster/software \
      --bind "${HOST_NCCL_PATH}:/opt/nccl" \
      --bind "${HOST_AWS_OFI_NCCL_PATH}:/opt/aws-ofi-nccl/lib" \
      --bind "${HOST_LIBFABRIC_LIB_PATH}:/opt/libfabric/lib" \
      --bind "${HOST_LIBFABRIC_INCLUDE_PATH}:/opt/libfabric/include" \
      --bind "${HOST_CXI_LIB_PATH}:/opt/cxi" \
      "${APPTAINER_SIF}" bash -c run_training

echo
echo "--- Finished :) ---"
