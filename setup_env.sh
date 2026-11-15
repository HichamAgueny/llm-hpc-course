#!/bin/bash
# ==============================================================================
# Environment Setup Script for LLM-HPC-Course (Bergen Edition)
# ==============================================================================
# Initializes the course workspace by:
# 1. Loading centralized cluster configuration from course_env.sh
# 2. Creating structured results, caches, and per-lab log directories
# 3. Linking or copying pre-downloaded shared models, datasets, and container images
# 4. Setting executable permissions on all course launchers and scripts
# 5. Running check_env.sh for pre-flight verification
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 1. Load centralized configuration if available
if [ -f "${SCRIPT_DIR}/course_env.sh" ]; then
    source "${SCRIPT_DIR}/course_env.sh"
fi

COURSE_ACCOUNT="${COURSE_SLURM_ACCOUNT:-nn9997k}"
CLUSTER_ROOT="${CLUSTER_PROJECT_ROOT:-/cluster/projects/${COURSE_ACCOUNT}}"

# Target directory: default to PROJECT_DIR or current repository
TARGET_BASE="${PROJECT_DIR:-${SCRIPT_DIR}}"

# Source directory for pre-downloaded course assets (models, datasets, SIFs)
SOURCE_BASE="${SOURCE_BASE:-${CLUSTER_ROOT}/shared_course/llm-hpc-course}"

MODE="link"
if [ "$1" == "--copy" ]; then
    MODE="copy"
fi

echo "================================================================"
echo "    LLM HPC Course — Environment Initialization"
echo "================================================================"
echo "Cluster Account:  ${COURSE_ACCOUNT}"
echo "Target Workspace: ${TARGET_BASE}"
echo "Source Assets:    ${SOURCE_BASE}"
echo "Asset Linking:    ${MODE} (pass --copy to force full file copy)"
echo "----------------------------------------------------------------"

# 2. Create core results and cache structures
echo "[1/4] Initializing results hierarchy and cache folders..."
mkdir -p "${TARGET_BASE}/results/checkpoint_results"
mkdir -p "${TARGET_BASE}/results/profiling_results"
mkdir -p "${TARGET_BASE}/results/evaluation_results"
mkdir -p "${TARGET_BASE}/hf_cache"
mkdir -p "${TARGET_BASE}/.cache/vllm"
mkdir -p "${TARGET_BASE}/apptainer/tmp"

# 3. Create per-lab output/log directories
echo "[2/4] Ensuring all Slurm log directories exist..."
LAB_OUT_DIRS=(
    "day1_single_gpu/finetuning/lora_slurm/job_container/out"
    "day1_single_gpu/finetuning/lora_slurm/job_eessi/out"
    "day1_single_gpu/finetuning/qlora_slurm/job_container/out"
    "day1_single_gpu/finetuning/qlora_slurm/job_eessi/out"
    "day1_single_gpu/profiling/out"
    "day1_single_gpu/evaluation/out"
    "day2_multi_gpu/fine-tuning/ft_single_node/lora_slurm/job_container/out"
    "day2_multi_gpu/fine-tuning/ft_single_node/lora_slurm/job_eessi/out"
    "day2_multi_gpu/fine-tuning/ft_single_node/qlora_slurm/out"
    "day2_multi_gpu/fine-tuning/ft_single_node/qat-lora_slurm/out"
    "day2_multi_gpu/fine-tuning/ft_multi_node/lora_slurm/job_container/out"
    "day2_multi_gpu/fine-tuning/ft_multi_node/lora_slurm/job_eessi/out"
    "day2_multi_gpu/fine-tuning/ft_multi_node/qlora_slurm/out"
    "day2_multi_gpu/fine-tuning/ft_multi_node/qat-lora_slurm/out"
    "day2_multi_gpu/evaluation/out"
    "day2_multi_gpu/profiling/out"
    "day2_multi_gpu/quantization/quat_bitsandbytes/out"
    "day2_multi_gpu/quantization/quat_torchao/out"
    "day2_multi_gpu/quantization/quat_gptqmodel/out"
    "day2_multi_gpu/serving/single_device/vllm_container/out"
    "day2_multi_gpu/serving/single_device/vllm_eessi/out"
    "day2_multi_gpu/serving/multi_device/vllm_container/out"
    "day2_multi_gpu/serving/multi_device/vllm_eessi/out"
    "day2_multi_gpu/inference/out"
)

for sub in "${LAB_OUT_DIRS[@]}"; do
    mkdir -p "${TARGET_BASE}/${sub}"
done

# 4. Link or Copy Shared Resources (Models, Datasets, and Apptainer Images)
echo "[3/4] Linking / copying shared resources..."
mkdir -p "${TARGET_BASE}/apptainer"

if [ -d "${SOURCE_BASE}/shared" ]; then
    if [ "$MODE" == "copy" ]; then
        echo " -> Copying shared/ models and datasets..."
        cp -r "${SOURCE_BASE}/shared" "${TARGET_BASE}/"
    else
        echo " -> Symlinking shared/ models and datasets (instant, saves quota)..."
        ln -sfn "${SOURCE_BASE}/shared" "${TARGET_BASE}/shared"
    fi
else
    echo " [INFO] Source shared directory not found at: ${SOURCE_BASE}/shared"
    echo "        (Will use existing ${TARGET_BASE}/shared if already present)"
fi

SIF_FILES=(
    "pytorch_25.08_cuda13.0_arm_custom.sif"
    "vllm0.12_cu131_py3.12_arm_custom.sif"
)

for sif in "${SIF_FILES[@]}"; do
    src_sif="${SOURCE_BASE}/apptainer/${sif}"
    dst_sif="${TARGET_BASE}/apptainer/${sif}"
    if [ -f "${src_sif}" ] && [ ! -f "${dst_sif}" ]; then
        if [ "$MODE" == "copy" ]; then
            echo " -> Copying ${sif}..."
            cp "${src_sif}" "${dst_sif}"
        else
            echo " -> Symlinking ${sif}..."
            ln -sfn "${src_sif}" "${dst_sif}"
        fi
    fi
done

# 5. Ensure All Scripts Have Executable Permissions
echo "[4/4] Setting execution permissions on all shell scripts..."
find "${TARGET_BASE}" -maxdepth 4 -name "*.sh" -exec chmod +x {} + 2>/dev/null || true
[ -f "${TARGET_BASE}/menu.sh" ] && chmod +x "${TARGET_BASE}/menu.sh"
[ -f "${TARGET_BASE}/run.sh" ] && chmod +x "${TARGET_BASE}/run.sh"
[ -f "${TARGET_BASE}/check_env.sh" ] && chmod +x "${TARGET_BASE}/check_env.sh"

echo "----------------------------------------------------------------"
echo "Setup Complete! Running pre-flight verification..."
echo "----------------------------------------------------------------"
if [ -f "${TARGET_BASE}/check_env.sh" ]; then
    bash "${TARGET_BASE}/check_env.sh"
fi
