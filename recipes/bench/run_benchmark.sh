#!/bin/bash -e

export PYTHONNOUSERSITE=1
export NUMEXPR_MAX_THREADS=72

echo "=== vLLM Benchmark Script (Fully Offline & Local) ==="
echo

# =====================================================================
# --- PATH CONFIGURATION ---
# =====================================================================
export CLUSTER_PROJECT_ROOT="/cluster/projects/nn9997k"
export PROJECT_DIR="${CLUSTER_PROJECT_ROOT}/${USER}/llm-hpc-course"
export RESULT_DIR="${PROJECT_DIR}/results"

# Apptainer Container Configuration
export CONTAINER_DIR="${PROJECT_DIR}/apptainer"
export APPTAINER_SIF="${CONTAINER_DIR}/vllm0.12_cu131_py3.12_arm_custom.sif"

# Connection file generated dynamically by serve_vllm.sh
export CONNECTION_FILE="${PROJECT_DIR}/day2_multi_gpu/serving/single_device/connection.env"

# Benchmark Output Directory & Filename
export BENCHMARK_RESULT_DIR="${RESULT_DIR}/benchmark_logs"
mkdir -p "${BENCHMARK_RESULT_DIR}"
export BENCHMARK_FILENAME="vllm_benchmark_results.json"
# =====================================================================

# Load active connection, model, and local tokenizer info from the server
if [[ -f "${CONNECTION_FILE}" ]]; then
    source "${CONNECTION_FILE}"
    echo "--Loaded connection info from: ${CONNECTION_FILE}"
else
    echo "ERROR: Connection file not found at ${CONNECTION_FILE}."
    echo "Make sure serve_vllm.sh is currently running!"
    exit 1
fi

# Ensure BASE_URL is correctly formatted for vllm bench
BASE_URL="${HOST}:${PORT}"

echo "----------------------------------------"
echo "Benchmark Configuration:"
echo "  Target Server URL:  ${BASE_URL}"
echo "  Active Model Tag:   ${MODEL}"
echo "  Local Tokenizer:    ${BASE_MODEL_DIR}"
echo "  Result Directory:   ${BENCHMARK_RESULT_DIR}"
echo "----------------------------------------"
echo

# =====================================================================
# Run vllm bench serve inside Apptainer (completely offline)
# =====================================================================
echo "--- Starting local vLLM benchmark execution ---"

apptainer exec --nv \
      -B "${PROJECT_DIR}:/workspace" \
      -B "${CLUSTER_PROJECT_ROOT}" \
      --pwd "/workspace" \
      "${APPTAINER_SIF}" \
      vllm bench serve \
          --backend openai-chat \
          --model "${MODEL}" \
          --tokenizer "${BASE_MODEL_DIR}" \
          --base-url "${BASE_URL}" \
          --endpoint /v1/chat/completions \
          --dataset-name random \
          --num-prompts 10 \
          --random-input-len 128 \
          --random-output-len 128 \
          --request-rate inf \
          --save-result \
          --save-detailed \
          --result-dir "${BENCHMARK_RESULT_DIR}" \
          --result-filename "${BENCHMARK_FILENAME}"

echo
echo "--- Benchmark completed successfully! ---"
echo "Results saved to: ${BENCHMARK_RESULT_DIR}/${BENCHMARK_FILENAME}"
