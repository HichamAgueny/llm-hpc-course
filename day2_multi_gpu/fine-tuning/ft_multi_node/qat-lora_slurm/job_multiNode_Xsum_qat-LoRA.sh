#!/bin/bash -e
#SBATCH --job-name=ft-llama3-1B-qat-lora-8gpu_xsum
#SBATCH --account=nn9997k
#SBATCH --time=00:15:00
#SBATCH --partition=accel
#SBATCH --nodes=2
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH -o ./out/%x-%j.out
#SBATCH --mem-per-gpu=97G

echo "--Node: $(hostname)"
echo
module load NRIS/GPU
module load NCCL/2.30.4-GCCcore-14.3.0-CUDA-13.0.0

# --- Variables and Paths (HOST-SIDE) ---
PROJECT_DIR="/cluster/projects/nn9997k"
MyWD="$PROJECT_DIR/$USER/llm-hpc-course"
MyCurrentWD="${MyWD}/day2_multi_gpu/fine-tuning/ft_multi_node/lora_slurm"
CONTAINER_DIR="${MyWD}/apptainer"
APPTAINER_SIF="${CONTAINER_DIR}/pytorch_25.08_cuda13.0_arm_custom.sif"

# Configs and python files for fine-tuning
CONFIG_FILE="${MyWD}/configs/qat-lora/llama3_2_1B_qat-lora_multi_node_XSum.yaml"
PYTHON_FILE="${MyWD}/recipes/distributed/qat_lora_finetune_distributed.py"

echo "--- My Main Directory (host): ${MyWD}"
echo "--- Bind-mounted inside container as: /workspace"
echo

# --- Slurm settings ---
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "--nbr of nodes: $SLURM_JOB_NUM_NODES"
echo "--nbr of GPUs per node: $SLURM_GPUS_PER_NODE"
echo

# --- Host-side library paths ---
export HOST_NCCL_PATH="${NCCL_HOME:-/cluster/software/NRIS/neoverse_v2/software/NCCL/2.30.4-GCCcore-14.3.0-CUDA-13.0.0}"
export HOST_AWS_OFI_NCCL_PATH="${EBROOTAWSMINOFIMINNCCL:-/cluster/software/NRIS/neoverse_v2/software/aws-ofi-nccl/1.19.1-GCCcore-14.3.0-CUDA-13.0.0}/lib"
export HOST_LIBFABRIC_LIB_PATH="/opt/cray/libfabric/2.3.1/lib64"
export HOST_LIBFABRIC_INCLUDE_PATH="/opt/cray/libfabric/2.3.1/include"
export HOST_CXI_LIB_PATH="/usr/lib64"

# --- Resolve head node ---
nodes=( $(scontrol show hostnames $SLURM_JOB_NODELIST) )
head_node_ip=$(getent hosts "${nodes[0]}" | awk '{print $1}')

export MASTER_ADDR="$head_node_ip"
export MASTER_PORT=25900
export WORLD_SIZE=$SLURM_NTASKS
export LOCAL_WORLD_SIZE=$SLURM_NTASKS_PER_NODE

echo "--WORLD_SIZE: $WORLD_SIZE"
echo "--LOCAL_WORLD_SIZE: $LOCAL_WORLD_SIZE"
echo

# --- Create the Inner Script (runs INSIDE container) ---
INNER_SCRIPT_TEMP="./.my_script_temp_${SLURM_JOB_ID}"
cat > "${INNER_SCRIPT_TEMP}" << EOF
#!/bin/bash -e

export PYTHONNOUSERSITE=1
export USE_FLASH_ATTENTION=1
#export NCCL_DEBUG=INFO

# --- Load Network Environments ---
if [ -f "${MyCurrentWD}/nccl_env.sh" ]; then
    source "${MyCurrentWD}/nccl_env.sh"
else
    echo "ERROR: ${MyCurrentWD}/nccl_env.sh not found!"
    exit 1
fi

# FORCE HOST NCCL: LD_PRELOAD to guarantee overriding internal PyTorch NCCL
#export LD_PRELOAD="/opt/nccl/lib/libnccl.so.2:\${LD_PRELOAD}"
export LD_LIBRARY_PATH="/opt/nccl/lib:/opt/aws-ofi-nccl/lib:/opt/libfabric/lib:/opt/cxi:\${LD_LIBRARY_PATH}"

echo "Checking NCCL library source:"
ldd \$(python -c "import torch; print(torch.__path__[0])")/lib/libtorch_cuda.so | grep nccl || echo "NCCL linked elsewhere"

export RANK=\$SLURM_PROCID
export LOCAL_RANK=\$SLURM_LOCALID

echo "RANK: \${RANK}, LOCAL_RANK: \${LOCAL_RANK}, WORLD_SIZE=$WORLD_SIZE, CUDA_VISIBLE_DEVICES: \${CUDA_VISIBLE_DEVICES}"

python "${PYTHON_FILE}" --config "${CONFIG_FILE}"
EOF

chmod +x "${INNER_SCRIPT_TEMP}"

# --- Suppress LMOD Debugging & Locale ---
export LMOD_SH_DBG_ON=0
export LANG=en_US.UTF-8
export LC_ALL=en_US.UTF-8

# --- Diagnostics ---
echo "--- Diagnostics ---"
echo "PATH: $PATH"
which apptainer || { echo "ERROR: apptainer not found"; exit 1; }
echo "NCCL module: $(module list 2>&1 | grep NCCL || echo 'NOT LOADED')"
echo "HOST_NCCL_PATH: $HOST_NCCL_PATH"
echo "aws-ofi-nccl path: $HOST_AWS_OFI_NCCL_PATH"
echo "SIF: $(ls -lh "${APPTAINER_SIF}" 2>/dev/null || echo MISSING)"
echo

echo "--- Launching the application inside Apptainer ---"

# --- Execute with Apptainer ---
time srun --network=disable_rdzv_get apptainer exec --nv \
      --bind "${MyWD}:/workspace" \
      --bind $PROJECT_DIR \
      --bind /cluster/software \
      --bind $HOST_NCCL_PATH:/opt/nccl \
      --bind $HOST_AWS_OFI_NCCL_PATH:/opt/aws-ofi-nccl/lib \
      --bind $HOST_LIBFABRIC_LIB_PATH:/opt/libfabric/lib \
      --bind $HOST_LIBFABRIC_INCLUDE_PATH:/opt/libfabric/include \
      --bind $HOST_CXI_LIB_PATH:/opt/cxi \
      "${APPTAINER_SIF}" \
      "${INNER_SCRIPT_TEMP}"

rm -f "${INNER_SCRIPT_TEMP}"
echo "--- Finished :) ---"
