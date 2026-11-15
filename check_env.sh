#!/bin/bash
# ==============================================================================
# Pre-Flight Environment Health Check
# LLM Fine-Tuning & Inference on HPC
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "${SCRIPT_DIR}/course_env.sh" ]; then
    source "${SCRIPT_DIR}/course_env.sh"
fi

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m' # No Color

echo -e "${BOLD}${BLUE}======================================================${NC}"
echo -e "${BOLD}${BLUE}    LLM HPC Course — Pre-Flight Environment Check     ${NC}"
echo -e "${BOLD}${BLUE}======================================================${NC}"
echo "Current User: ${USER}"
echo "Host:         $(hostname)"
echo "Date:         $(date)"
echo ""

ERRORS=0
WARNINGS=0

check_pass() {
    echo -e "  [ ${GREEN}PASS${NC} ] $1"
}

check_fail() {
    echo -e "  [ ${RED}FAIL${NC} ] $1"
    ERRORS=$((ERRORS + 1))
}

check_warn() {
    echo -e "  [ ${YELLOW}WARN${NC} ] $1"
    WARNINGS=$((WARNINGS + 1))
}

# 1. Check Slurm Environment
echo -e "${BOLD}1. Checking Slurm Tools & Cluster Commands...${NC}"
if command -v sbatch >/dev/null 2>&1 && command -v squeue >/dev/null 2>&1; then
    check_pass "Slurm commands (sbatch, squeue) are available"
else
    check_warn "Slurm commands not found in PATH (are you running on a cluster login node?)"
fi

# 2. Check Apptainer
echo -e "\n${BOLD}2. Checking Container Runtime (Apptainer)...${NC}"
if command -v apptainer >/dev/null 2>&1; then
    APPTAINER_VER=$(apptainer --version 2>/dev/null || echo "unknown")
    check_pass "Apptainer is installed (${APPTAINER_VER})"
else
    check_warn "Apptainer command not found in PATH (load module or ensure it is installed)"
fi

# 3. Check Base Project Workspace
echo -e "\n${BOLD}3. Checking Course Workspaces & Directories...${NC}"
if [ -d "${PROJECT_DIR}" ]; then
    check_pass "Course project directory exists: ${PROJECT_DIR}"
else
    check_warn "Course directory not found at: ${PROJECT_DIR}"
fi

# 4. Check Shared Resources
echo -e "\n${BOLD}4. Checking Shared Models & Datasets...${NC}"
if [ -d "${BASE_MODEL_DIR}" ]; then
    check_pass "Base LLaMA model directory found: ${BASE_MODEL_DIR}"
else
    check_warn "Base model directory not found at: ${BASE_MODEL_DIR}"
fi

if [ -f "${QA_DATASET_DIR}" ]; then
    check_pass "Question-Answering dataset found: ${QA_DATASET_DIR}"
else
    check_warn "QA dataset not found at: ${QA_DATASET_DIR}"
fi

# 5. Check Container Images
echo -e "\n${BOLD}5. Checking Apptainer SIF Images...${NC}"
if [ -f "${APPTAINER_SIF}" ]; then
    check_pass "PyTorch custom training image found: $(basename "${APPTAINER_SIF}")"
else
    check_warn "PyTorch container image not found at: ${APPTAINER_SIF}"
fi

if [ -f "${VLLM_SIF}" ]; then
    check_pass "vLLM custom serving image found: $(basename "${VLLM_SIF}")"
else
    check_warn "vLLM container image not found at: ${VLLM_SIF}"
fi

# 6. Prepare / Verify Output Directories
echo -e "\n${BOLD}6. Verifying & Initializing Result Directories...${NC}"
for dir in \
    "${SCRIPT_DIR}/results/checkpoint_results" \
    "${SCRIPT_DIR}/results/profiling_results" \
    "${SCRIPT_DIR}/results/evaluation_results" \
    "${SCRIPT_DIR}/day1_single_gpu/finetuning/lora_slurm/job_container/out" \
    "${SCRIPT_DIR}/day1_single_gpu/finetuning/lora_slurm/job_eessi/out" \
    "${SCRIPT_DIR}/day1_single_gpu/finetuning/qlora_slurm/job_container/out" \
    "${SCRIPT_DIR}/day1_single_gpu/finetuning/qlora_slurm/job_eessi/out" \
    "${SCRIPT_DIR}/day1_single_gpu/profiling/out" \
    "${SCRIPT_DIR}/day1_single_gpu/evaluation/out" \
    "${SCRIPT_DIR}/day2_multi_gpu/fine-tuning/ft_single_node/lora_slurm/job_container/out" \
    "${SCRIPT_DIR}/day2_multi_gpu/fine-tuning/ft_single_node/lora_slurm/job_eessi/out" \
    "${SCRIPT_DIR}/day2_multi_gpu/fine-tuning/ft_single_node/qlora_slurm/out" \
    "${SCRIPT_DIR}/day2_multi_gpu/fine-tuning/ft_single_node/qat-lora_slurm/out" \
    "${SCRIPT_DIR}/day2_multi_gpu/fine-tuning/ft_multi_node/lora_slurm/job_container/out" \
    "${SCRIPT_DIR}/day2_multi_gpu/fine-tuning/ft_multi_node/lora_slurm/job_eessi/out" \
    "${SCRIPT_DIR}/day2_multi_gpu/fine-tuning/ft_multi_node/qlora_slurm/out" \
    "${SCRIPT_DIR}/day2_multi_gpu/fine-tuning/ft_multi_node/qat-lora_slurm/out" \
    "${SCRIPT_DIR}/day2_multi_gpu/evaluation/out" \
    "${SCRIPT_DIR}/day2_multi_gpu/profiling/out" \
    "${SCRIPT_DIR}/day2_multi_gpu/inference/out" \
    "${SCRIPT_DIR}/day2_multi_gpu/serving/single_device/vllm_container/out" \
    "${SCRIPT_DIR}/day2_multi_gpu/serving/multi_device/vllm_container/out"
do
    if [ ! -d "$dir" ]; then
        mkdir -p "$dir" 2>/dev/null
    fi
done
check_pass "Output folders and results directory hierarchy are prepared"

echo -e "\n${BOLD}${BLUE}======================================================${NC}"
if [ $ERRORS -eq 0 ] && [ $WARNINGS -eq 0 ]; then
    echo -e "${GREEN}${BOLD}✓ Pre-flight check PASSED! You are ready to run jobs.${NC}"
elif [ $ERRORS -eq 0 ]; then
    echo -e "${YELLOW}${BOLD}⚠ Pre-flight check completed with ${WARNINGS} warning(s).${NC}"
    echo -e "Review warnings above. (On a login node, local model paths will resolve once on Olivia)."
else
    echo -e "${RED}${BOLD}✗ Pre-flight check found ${ERRORS} error(s). Please resolve them before submitting jobs.${NC}"
fi
echo -e "${BOLD}${BLUE}======================================================${NC}"

