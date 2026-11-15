#!/bin/bash -e

export PYTHONNOUSERSITE=1
export NUMEXPR_MAX_THREADS=72

echo "=== vLLM Benchmark Script (Fully Offline & Local) ==="
echo

# =====================================================================
# --- PATH CONFIGURATION ---
# =====================================================================
# Paths and container come from course_env.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/course_env.sh"

# Connection file written by the running server (serve_vllm*.sh).
# Default: the single-GPU container server; override with CONNECTION_FILE=...
export CONNECTION_FILE="${CONNECTION_FILE:-${COURSE_ROOT}/day2_multi_gpu/serving/single_device/vllm_container/connection.env}"

# Benchmark Output Directory & Filename
export BENCHMARK_RESULT_DIR="${BENCHMARK_RESULTS_DIR}"
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

apptainer exec --nv ${CONTAINER_BINDS} \
      --pwd "/workspace" \
      "${VLLM_SIF}" \
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
