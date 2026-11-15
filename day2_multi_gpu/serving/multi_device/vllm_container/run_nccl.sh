
#!/bin/bash
set -e

# ==========================================================
# 1. Project paths
# ==========================================================

export CLUSTER_PROJECT_ROOT="/cluster/work/projects/nn9971k"

export PROJECT_DIR="${CLUSTER_PROJECT_ROOT}/${USER}/llm-hpc-course"

export CONTAINER_DIR="${PROJECT_DIR}/apptainer"

export APPTAINER_SIF="${CONTAINER_DIR}/vllm0.12_cu131_py3.12_arm_custom.sif"

TEST_SCRIPT="/workspace/day2_multi_gpu/serving/multi_device/vllm_container/test_nccl.py"

# ==========================================================
# 2. GPU configuration
# ==========================================================

NUM_GPUS="${SLURM_GPUS_ON_NODE:-4}"

echo "=========================================="
echo "       NCCL Interconnect Benchmark"
echo "=========================================="

echo "Project:   ${PROJECT_DIR}"
echo "Container: ${APPTAINER_SIF}"
echo "GPUs:      ${NUM_GPUS}"
echo "=========================================="

# ==========================================================
# 3. Validate configuration
# ==========================================================

if [ ! -f "${APPTAINER_SIF}" ]; then
    echo "ERROR: Container image not found: ${APPTAINER_SIF}"
    exit 1
fi

if [ ! -f "${PROJECT_DIR}/day2_multi_gpu/serving/multi_device/vllm_container/test_nccl.py" ]; then
    echo "ERROR: NCCL benchmark script not found."
    exit 1
fi

if [ "${NUM_GPUS}" -lt 2 ]; then
    echo "ERROR: At least 2 GPUs are required."
    exit 1
fi

# ==========================================================
# 4. Run NCCL benchmark inside Apptainer
# ==========================================================

apptainer exec --nv \
    --bind "${PROJECT_DIR}:/workspace" \
    --pwd "/workspace" \
    "${APPTAINER_SIF}" \
    torchrun \
        --standalone \
        --nnodes=1 \
        --nproc_per_node="${NUM_GPUS}" \
        "${TEST_SCRIPT}"

