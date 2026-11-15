#!/bin/bash -e

export PYTHONNOUSERSITE=1
export NUMEXPR_MAX_THREADS=72

export NUM_GPUS="${SLURM_GPUS_ON_NODE:-4}"
echo "--Node: $(hostname)"
echo "--Allocated GPUs: ${NUM_GPUS}"
echo

# =====================================================================
# --- CONFIGURABLE VARIABLES AND PATHS (USERS ONLY MODIFY THIS BLOCK) ---
# =====================================================================

# 1. Base Project Allocation
export CLUSTER_PROJECT_ROOT="/cluster/projects/nn9997k"

# 2. Personal Workspace Setup
export PROJECT_DIR="${CLUSTER_PROJECT_ROOT}/${USER}/llm-hpc-course"
export SHARED_DIR="${PROJECT_DIR}/shared"
export RESULT_DIR="${PROJECT_DIR}/results"

# 3. Model, Quantization, Adapter & Serving Parameters
export BASE_MODEL_DIR="${MODEL_PATH:-${SHARED_DIR}/models/Llama-3.2-1B-Instruct}"
# Set quantization method for GPTQModel
export QUANTIZED_MODEL_DIR="${QUANTIZED_MODEL_PATH:-${SHARED_DIR}/models/Llama-3.2-1B-Instruct-gptq_4bit}"

# Serve the GPTQ quantized model as the base weight layer
export ACTIVE_MODEL_DIR="${QUANTIZED_MODEL_DIR}"

# LoRA Adapter path
export LORA_ADAPTER_DIR="${LORA_PATH-${RESULT_DIR}/checkpoint_results/llama3.2-1b_lora_single_device/checkpoints/epoch_0}"

export QUANTIZATION="${QUANTIZATION:-gptq_marlin}"
export MAX_LORA_RANK="${MAX_LORA_RANK:-64}"
export PORT="${PORT:-8000}"

# 4. Output / Connection File Paths
export CONNECTION_DIR="${PROJECT_DIR}/day2_multi_gpu/serving/multi_device/vllm_container"
mkdir -p "${CONNECTION_DIR}"
export CONNECTION_FILE="${CONNECTION_DIR}/connection.env"

# 5. Apptainer Container Configuration
export CONTAINER_DIR="${PROJECT_DIR}/apptainer"
export APPTAINER_SIF="${CONTAINER_DIR}/vllm0.12_cu131_py3.12_arm_custom.sif"

# 6. Caches under project share
export HF_HOME="${PROJECT_DIR}/hf_cache"
export VLLM_CACHE_ROOT="${PROJECT_DIR}/.cache/vllm"
mkdir -p "${HF_HOME}" "${VLLM_CACHE_ROOT}"
# =====================================================================

# -------- LoRA Detection --------
if [[ -n "${LORA_ADAPTER_DIR}" ]]; then
    USE_LORA=true
    MODEL_NAME="custom_lora"
else
    USE_LORA=false
    MODEL_NAME="${ACTIVE_MODEL_DIR}"
fi

echo "--- My Main Directory (host): ${PROJECT_DIR}"
echo "--- Bind-mounted inside container as: /workspace"
echo
echo "=== vLLM TorchAO + LoRA Server Configuration ==="
echo "Quantized Base Model: ${ACTIVE_MODEL_DIR}"
echo "Local Tokenizer Src:  ${BASE_MODEL_DIR}"
echo "Quantization Method:  ${QUANTIZATION}"
echo "LORA_ADAPTER_DIR:     ${LORA_ADAPTER_DIR:-disabled}"
echo "GPUs:                 ${NUM_GPUS}"
echo "Status:               Starting API Server on Port ${PORT}..."
echo "================================="
echo "💡 Usage Reminder:"
echo "   • Run with LoRA with quantized base model(default):  ./serve_vllm_quantized.sh"
echo "   • Run Quantized Base Model only:      LORA_PATH=\"\" ./serve_vllm_quantized.sh"
echo "================================="
echo

# =====================================================================
# STEP 1: Validation Sequence
# =====================================================================
echo "--- Validating model, tokenizer, and adapter paths ---"

if [[ ! -d "${ACTIVE_MODEL_DIR}" ]]; then
    echo "ERROR: TorchAO quantized model not found at: ${ACTIVE_MODEL_DIR}"
    exit 1
fi

if [[ ! -d "${BASE_MODEL_DIR}" ]]; then
    echo "ERROR: Base model (tokenizer fallback) not found at: ${BASE_MODEL_DIR}"
    exit 1
fi

if [[ "${USE_LORA}" == true ]]; then
    if [[ ! -d "${LORA_ADAPTER_DIR}" ]]; then
        echo "ERROR: LoRA checkpoint directory not found at: ${LORA_ADAPTER_DIR}"
        exit 1
    fi
    if [[ ! -f "${LORA_ADAPTER_DIR}/adapter_config.json" ]]; then
        echo "ERROR: adapter_config.json not found in: ${LORA_ADAPTER_DIR}"
        exit 1
    fi
fi
echo "--Validation passed :)"
echo

# =====================================================================
# STEP 2: Write Connection Info for Client & Benchmarking Scripts
# =====================================================================
echo "--- Writing connection info ---"
cat <<EOF > "${CONNECTION_FILE}"
HOST=http://$(hostname)
PORT=${PORT}
MODEL=${MODEL_NAME}
BASE_MODEL_DIR=${BASE_MODEL_DIR}
EOF

echo "--Connection info written to: ${CONNECTION_FILE}"
echo "--Host: http://$(hostname):${PORT}"
echo "--Model tag for client/benchmark: ${MODEL_NAME}"
echo

# =====================================================================
# STEP 3: Build & Launch vLLM API Server Inside Apptainer
# =====================================================================
VLLM_ARGS=(
    "serve" "${ACTIVE_MODEL_DIR}"
    "--tensor-parallel-size" "${NUM_GPUS}"
    "--host" "0.0.0.0"
    "--port" "${PORT}"
    "--quantization" "${QUANTIZATION}"
)

if [[ "${USE_LORA}" == true ]]; then
    VLLM_ARGS+=(
        "--enable-lora"
        "--lora-modules" "custom_lora=${LORA_ADAPTER_DIR}"
        "--max-lora-rank" "${MAX_LORA_RANK}"
    )
fi

echo "--- Launching TorchAO + LoRA vLLM Server inside Apptainer ---"

apptainer exec --nv \
      -B "${PROJECT_DIR}:/workspace" \
      -B "${CLUSTER_PROJECT_ROOT}" \
      --pwd "/workspace" \
      "${APPTAINER_SIF}" \
      vllm "${VLLM_ARGS[@]}"
