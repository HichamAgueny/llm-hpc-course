#!/bin/bash

# =========================================================================
# EDIT PATHS HERE (ONLY ONCE)
# =========================================================================
IMAGE_PATH="/cluster/projects/nn9997k/${USER}/llm-hpc-course/apptainer/pytorch_25.08_cuda13.0_arm_custom.sif"
WORKSPACE_DIR="/cluster/projects/nn9997k/${USER}/llm-hpc-course"
PROJECTS_DIR="/cluster/projects/nn9997k/${USER}/llm-hpc-course"

# Host Slingshot/Network Paths
HOST_NCCL="/cluster/software/NRIS/neoverse_v2/software/NCCL/2.30.4-GCCcore-14.3.0-CUDA-13.0.0"
HOST_AWS_OFI="/cluster/software/NRIS/neoverse_v2/software/aws-ofi-nccl/1.19.1-GCCcore-14.3.0-CUDA-13.0.0/lib"
HOST_FABRIC="/opt/cray/libfabric/2.3.1/lib64"
HOST_CXI="/usr/lib64"
# =========================================================================

MODE=$1
shift # Pass everything else (python, flags) to the container

# Base binds used by every single job
BINDS="--bind ${WORKSPACE_DIR}:/workspace --bind ${PROJECTS_DIR}:/projects_workspace"

# If the job is multinode, append the network libraries and environment variables
if [ "$MODE" == "multinode" ]; then
    # Linker Paths
    export APPTAINERENV_LD_LIBRARY_PATH="/opt/hpc-libs/aws-ofi-nccl:/opt/hpc-libs/nccl/lib:/opt/hpc-libs/libfabric/lib:/opt/hpc-libs/cxi:\$LD_LIBRARY_PATH"
    
    # HPE Slingshot / Cray Fabric Performance Variables (from nccl_env.sh)
    export APPTAINERENV_HSA_FORCE_FINE_GRAIN_PCIE=1
    export APPTAINERENV_FI_MR_CACHE_MONITOR="userfaultfd"
    export APPTAINERENV_FI_CXI_DISABLE_HOST_REGISTER=1
    export APPTAINERENV_FI_CXI_DEFAULT_CQ_SIZE=131072
    export APPTAINERENV_FI_CXI_RDZV_PROTO="alt_read"
    export APPTAINERENV_FI_CXI_RDZV_EAGER_SIZE=0
    export APPTAINERENV_FI_CXI_RDZV_THRESHOLD=0
    export APPTAINERENV_FI_CXI_RDZV_GET_MIN=0
    export APPTAINERENV_FI_CXI_DEFAULT_TX_SIZE=2048
    export APPTAINERENV_NCCL_CROSS_NIC=1
    export APPTAINERENV_NCCL_NET_GDR_LEVEL="PHB"
    export APPTAINERENV_NCCL_SOCKET_IFNAME="hsn0,hsn1,hsn2,hsn3"
    export APPTAINERENV_NCCL_NET="AWS Libfabric"
    export APPTAINERENV_FI_CXI_RX_MATCH_MODE="hybrid"
    export APPTAINERENV_FI_PROVIDER="cxi"
    
    # Mount host fabric paths into the container
    BINDS="${BINDS} --bind ${HOST_NCCL}:/opt/hpc-libs/nccl --bind ${HOST_AWS_OFI}:/opt/hpc-libs/aws-ofi-nccl --bind ${HOST_FABRIC}:/opt/hpc-libs/libfabric/lib --bind ${HOST_CXI}:/opt/hpc-libs/cxi"
fi

# Single point of execution
exec apptainer exec --nv ${BINDS} "${IMAGE_PATH}" "$@"
