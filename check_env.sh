#!/bin/bash
# ==============================================================================
# Pre-Flight Environment Health Check
# LLM Fine-Tuning & Inference on HPC
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/course_env.sh"

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m' # No Color

echo -e "${BOLD}${BLUE}======================================================${NC}"
echo -e "${BOLD}${BLUE}    LLM HPC Course — Pre-Flight Environment Check     ${NC}"
echo -e "${BOLD}${BLUE}======================================================${NC}"
echo "Current User:  ${USER}"
echo "Host:          $(hostname)"
echo "Date:          $(date)"
echo "Course folder: ${COURSE_ROOT}"
echo "Slurm account: ${COURSE_SLURM_ACCOUNT}   reservation: ${COURSE_SLURM_RESERVATION:-<none>}"
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

# 1. Slurm
echo -e "${BOLD}1. Checking Slurm Tools & Cluster Commands...${NC}"
if command -v sbatch >/dev/null 2>&1 && command -v squeue >/dev/null 2>&1; then
    check_pass "Slurm commands (sbatch, squeue) are available"
    if command -v scontrol >/dev/null 2>&1 && [ -n "${COURSE_SLURM_RESERVATION}" ]; then
        if scontrol show reservation "${COURSE_SLURM_RESERVATION}" >/dev/null 2>&1; then
            check_pass "Reservation '${COURSE_SLURM_RESERVATION}' exists"
        else
            check_warn "Reservation '${COURSE_SLURM_RESERVATION}' not found (expired?). Set COURSE_SLURM_RESERVATION=\"\" in course_env.sh to submit without it"
        fi
    fi
else
    check_warn "Slurm commands not found in PATH (are you running on a cluster login node?)"
fi

# 2. Apptainer
echo -e "\n${BOLD}2. Checking Container Runtime (Apptainer)...${NC}"
if command -v apptainer >/dev/null 2>&1; then
    APPTAINER_VER=$(apptainer --version 2>/dev/null || echo "unknown")
    check_pass "Apptainer is installed (${APPTAINER_VER})"
else
    check_warn "Apptainer command not found in PATH (load module or ensure it is installed)"
fi

# 3. Shared assets
echo -e "\n${BOLD}3. Checking Shared Course Assets...${NC}"
if [ -d "${COURSE_ASSETS_DIR}" ]; then
    check_pass "Shared course assets folder: ${COURSE_ASSETS_DIR}"
else
    check_fail "Shared course assets folder not found: ${COURSE_ASSETS_DIR} (check COURSE_ASSETS_DIR in course_env.sh)"
fi

if [ -L "${COURSE_ROOT}/shared" ] || [ -d "${COURSE_ROOT}/shared" ]; then
    check_pass "Course 'shared' folder: ${COURSE_ROOT}/shared"
else
    check_warn "No ${COURSE_ROOT}/shared folder yet (run ./setup_env.sh) - using ${SHARED_DIR} directly"
fi

# 4. Model & dataset
echo -e "\n${BOLD}4. Checking Shared Models & Datasets...${NC}"
if [ -d "${BASE_MODEL_DIR}" ]; then
    check_pass "Base model: ${BASE_MODEL_DIR}"
else
    check_fail "Base model not found: ${BASE_MODEL_DIR}"
fi

if [ -f "${QA_DATASET_DIR}" ]; then
    check_pass "QA dataset: ${QA_DATASET_DIR}"
else
    check_fail "QA dataset not found: ${QA_DATASET_DIR}"
fi

# 5. Container images
echo -e "\n${BOLD}5. Checking Apptainer SIF Images...${NC}"
if [ -f "${PYTORCH_SIF}" ]; then
    check_pass "PyTorch training image: ${PYTORCH_SIF}"
else
    check_fail "PyTorch container image not found: ${PYTORCH_SIF}"
fi

if [ -f "${VLLM_SIF}" ]; then
    check_pass "vLLM serving image: ${VLLM_SIF}"
else
    check_fail "vLLM container image not found: ${VLLM_SIF}"
fi

# 6. Output folders
echo -e "\n${BOLD}6. Checking Output Folders...${NC}"
mkdir -p "${CHECKPOINT_RESULTS_DIR}" "${PROFILING_RESULTS_DIR}" "${EVAL_RESULTS_DIR}" \
         "${QUANTIZED_MODELS_DIR}" "${BENCHMARK_RESULTS_DIR}" 2>/dev/null
missing_out=0
while IFS= read -r job; do
    out_dir="$(dirname "${job}")/out"
    if [ ! -d "${out_dir}" ]; then
        mkdir -p "${out_dir}" 2>/dev/null || missing_out=$((missing_out + 1))
    fi
done < <(grep -rl --include='*.sh' '^#SBATCH' "${COURSE_ROOT}/day1_single_gpu" "${COURSE_ROOT}/day2_multi_gpu")
if [ ${missing_out} -eq 0 ] && [ -w "${RESULTS_ROOT}" ]; then
    check_pass "results/ folders and ./out/ Slurm log folders are ready"
else
    check_fail "Could not create some output folders (check write permissions in ${COURSE_ROOT})"
fi

# 7. Information: which LoRA adapter the eval / serving labs will use
echo -e "\n${BOLD}7. LoRA adapter used by evaluation / serving / inference...${NC}"
if [ -d "${LORA_ADAPTER_DIR}" ]; then
    check_pass "Adapter found: ${LORA_ADAPTER_DIR}"
else
    echo -e "  [ ${BLUE}INFO${NC} ] Not created yet (expected before the first fine-tuning): ${LORA_ADAPTER_DIR}"
fi

echo -e "\n${BOLD}${BLUE}======================================================${NC}"
if [ $ERRORS -eq 0 ] && [ $WARNINGS -eq 0 ]; then
    echo -e "${GREEN}${BOLD}✓ Pre-flight check PASSED! You are ready to run jobs.${NC}"
elif [ $ERRORS -eq 0 ]; then
    echo -e "${YELLOW}${BOLD}⚠ Pre-flight check completed with ${WARNINGS} warning(s).${NC}"
    echo -e "Review warnings above."
else
    echo -e "${RED}${BOLD}✗ Pre-flight check found ${ERRORS} error(s) and ${WARNINGS} warning(s).${NC}"
    echo -e "Fix the errors above before submitting jobs (on a laptop, cluster paths are expected to fail)."
fi
echo -e "${BOLD}${BLUE}======================================================${NC}"
