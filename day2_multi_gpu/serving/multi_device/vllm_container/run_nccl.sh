#!/bin/bash
set -e
# NCCL interconnect benchmark between the GPUs of one node.
# Run it inside an interactive 4-GPU shell:  bash srun_sinteractive_4gpus ; ./run_nccl.sh

# ==========================================================
# 1. Course settings (paths and container from course_env.sh)
# ==========================================================
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)/course_env.sh"

TEST_SCRIPT="/workspace/day2_multi_gpu/serving/multi_device/vllm_container/test_nccl.py"

# ==========================================================
# 2. GPU configuration
# ==========================================================

NUM_GPUS="${SLURM_GPUS_ON_NODE:-4}"

echo "=========================================="
echo "       NCCL Interconnect Benchmark"
echo "=========================================="

echo "Project:   ${COURSE_ROOT}"
echo "Container: ${VLLM_SIF}"
echo "GPUs:      ${NUM_GPUS}"
echo "=========================================="

# ==========================================================
# 3. Validate configuration
# ==========================================================

if [ ! -f "${VLLM_SIF}" ]; then
    echo "ERROR: Container image not found: ${VLLM_SIF}"
    exit 1
fi

if [ ! -f "${COURSE_ROOT}/day2_multi_gpu/serving/multi_device/vllm_container/test_nccl.py" ]; then
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

apptainer exec --nv ${CONTAINER_BINDS} \
    --pwd "/workspace" \
    "${VLLM_SIF}" \
    torchrun \
        --standalone \
        --nnodes=1 \
        --nproc_per_node="${NUM_GPUS}" \
        "${TEST_SCRIPT}"

