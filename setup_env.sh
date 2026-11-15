#!/bin/bash
# ==============================================================================
# Environment Setup Script for the LLM-HPC course
# ==============================================================================
# Run once after cloning:   ./setup_env.sh          (copies the shared assets)
#                           ./setup_env.sh --link   (links them instead: instant, saves quota)
#
# 1. Loads the course configuration from course_env.sh
# 2. Creates the results/ and cache folders
# 3. Creates the ./out/ Slurm log folder next to every job script
# 4. Copies (or links) the shared models, datasets and container images
#    from COURSE_ASSETS_DIR (see course_env.sh); files already present are kept
# 5. Makes all scripts executable and runs check_env.sh
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/course_env.sh"

MODE="copy"
case "${1:-}" in
    --link)    MODE="link" ;;
    --copy|"") MODE="copy" ;;
    *) echo "Usage: ./setup_env.sh [--link]"; exit 1 ;;
esac

echo "================================================================"
echo "    LLM HPC Course — Environment Initialization"
echo "================================================================"
echo "Slurm account:    ${COURSE_SLURM_ACCOUNT}"
echo "Reservation:      ${COURSE_SLURM_RESERVATION:-<none>}"
echo "Course folder:    ${COURSE_ROOT}"
echo "Shared assets:    ${COURSE_ASSETS_DIR}"
echo "Asset mode:       ${MODE} (default: copy; pass --link to link instead)"
echo "----------------------------------------------------------------"

# 1. Results and cache folders
echo "[1/4] Creating results and cache folders..."
mkdir -p "${CHECKPOINT_RESULTS_DIR}" "${PROFILING_RESULTS_DIR}" "${EVAL_RESULTS_DIR}" \
         "${QUANTIZED_MODELS_DIR}" "${BENCHMARK_RESULTS_DIR}" \
         "${HF_HOME}" "${VLLM_CACHE_ROOT}" "${COURSE_ROOT}/apptainer/tmp"

# 2. Slurm log folders: one ./out/ next to every script with #SBATCH lines
echo "[2/4] Creating ./out/ Slurm log folders next to every job script..."
n_out=0
while IFS= read -r job; do
    mkdir -p "$(dirname "${job}")/out"
    n_out=$((n_out + 1))
done < <(grep -rl --include='*.sh' '^#SBATCH' "${COURSE_ROOT}/day1_single_gpu" "${COURSE_ROOT}/day2_multi_gpu")
echo "      ${n_out} job scripts found"

# 3. Shared models, datasets and container images
echo "[3/4] Shared resources from ${COURSE_ASSETS_DIR} (${MODE})..."
if [ -d "${COURSE_ASSETS_DIR}/shared" ]; then
    if [ -e "${COURSE_ROOT}/shared" ] && [ ! -L "${COURSE_ROOT}/shared" ]; then
        echo " -> ${COURSE_ROOT}/shared already exists - keeping it (delete it to copy again)."
    elif [ "$MODE" == "copy" ]; then
        echo " -> Copying shared/ models and datasets (this can take a few minutes)..."
        rm -f "${COURSE_ROOT}/shared"
        cp -r "${COURSE_ASSETS_DIR}/shared" "${COURSE_ROOT}/shared"
    else
        echo " -> Linking shared/ models and datasets (instant, saves quota)..."
        ln -sfn "${COURSE_ASSETS_DIR}/shared" "${COURSE_ROOT}/shared"
    fi
else
    echo " [WARN] Shared folder not found: ${COURSE_ASSETS_DIR}/shared"
    echo "        Check COURSE_ASSETS_DIR in course_env.sh."
fi

for src_sif in "${COURSE_ASSETS_DIR}"/apptainer/*.sif; do
    [ -f "${src_sif}" ] || continue
    sif_name="$(basename "${src_sif}")"
    dst_sif="${COURSE_ROOT}/apptainer/${sif_name}"
    if [ -f "${dst_sif}" ] && [ ! -L "${dst_sif}" ]; then
        echo " -> ${sif_name} already present - keeping it."
        continue
    fi
    if [ "$MODE" == "copy" ]; then
        echo " -> Copying ${sif_name}..."
        rm -f "${dst_sif}"
        cp "${src_sif}" "${dst_sif}"
    else
        echo " -> Linking ${sif_name}..."
        ln -sfn "${src_sif}" "${dst_sif}"
    fi
done

# 4. Executable permissions
echo "[4/4] Setting execution permissions on all shell scripts..."
find "${COURSE_ROOT}" -name "*.sh" -not -path "*/llama.cpp/*" -exec chmod +x {} + 2>/dev/null || true

echo "----------------------------------------------------------------"
echo "Setup complete! Running pre-flight verification..."
echo "----------------------------------------------------------------"
bash "${COURSE_ROOT}/check_env.sh"

echo ""
echo "Next steps:"
echo "  ./run.sh day1 lora        # submit your first job (or: ./menu.sh)"
echo "  source course_env.sh      # once per login, if you submit with plain 'sbatch'"
