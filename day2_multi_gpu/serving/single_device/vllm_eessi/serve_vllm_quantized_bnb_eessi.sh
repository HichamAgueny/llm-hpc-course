# Start the server: run ./serve_vllm_quantized_bnb_eessi.sh inside an interactive GPU shell: bash srun_sinteractive)
# Then chat with it from a login node, in this folder:  ./chat.sh

export PYTHONNOUSERSITE=1

ml EESSI/2025.06
#module load torchtune/0.6.1-foss-2025b-PyTorch-2.12.0-CUDA-12.9.1
module load vllm/0.22.1-foss-2025b-PyTorch-2.12.0-CUDA-12.9.1
module load Transformers
#module load accelerate
module load bitsandbytes

export NUMEXPR_MAX_THREADS=72

export NUM_GPUS="${SLURM_GPUS_ON_NODE:-1}"
echo "--Node: $(hostname)"
echo "--Allocated GPUs: ${NUM_GPUS}"
echo

# =====================================================================
# --- CONFIGURABLE VARIABLES AND PATHS (USERS ONLY MODIFY THIS BLOCK) ---
# =====================================================================

# 1. Base Project Allocation
export CLUSTER_PROJECT_ROOT="/cluster/work/projects/nn9971k"

# 2. Personal Workspace Setup
export PROJECT_DIR="${CLUSTER_PROJECT_ROOT}/${USER}/llm-hpc-course"
export SHARED_DIR="${PROJECT_DIR}/shared"
export RESULT_DIR="${PROJECT_DIR}/results"

# 3. Model, Quantization, Adapter & Serving Parameters
export BASE_MODEL_DIR="${MODEL_PATH:-${SHARED_DIR}/models/Llama-3.2-1B-Instruct}"
# Set quantization method for bitsandbytes
export QUANTIZED_MODEL_DIR="${QUANTIZED_MODEL_PATH:-${SHARED_DIR}/models/Llama-3.2-1B-Instruct-bnb_8bit}"

# Serve the bitsandbytes quantized model as the base weight layer
export ACTIVE_MODEL_DIR="${QUANTIZED_MODEL_DIR}"

# LoRA Adapter path
export LORA_ADAPTER_DIR="${LORA_PATH-${RESULT_DIR}/checkpoint_results/eessi_llama3.2-1b_lora_single_device/checkpoints/epoch_1}"

export QUANTIZATION="${QUANTIZATION:-bitsandbytes}"
export MAX_LORA_RANK="${MAX_LORA_RANK:-64}"
export PORT="${PORT:-8000}"

# 4. Output / Connection File Paths
export CONNECTION_DIR="${PROJECT_DIR}/day2_multi_gpu/serving/single_device/vllm_eessi"
mkdir -p "${CONNECTION_DIR}"
export CONNECTION_FILE="${CONNECTION_DIR}/connection.env"

# 5. Caches under project share
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
echo
echo "=== vLLM BitsAndBytes + LoRA Server Configuration ==="
echo "Quantized Base Model: ${ACTIVE_MODEL_DIR}"
echo "Local Tokenizer Src:  ${BASE_MODEL_DIR}"
echo "Quantization Method:  ${QUANTIZATION}"
echo "LORA_ADAPTER_DIR:     ${LORA_ADAPTER_DIR:-disabled}"
echo "GPUs:                 ${NUM_GPUS}"
echo "Status:               Starting API Server on Port ${PORT}..."
echo "================================="
echo "💡 Usage Reminder:"
echo "   • Run with LoRA with quantized base model(default):  ./serve_vllm_quantized_bnb_eessi.sh"
echo "   • Run Quantized Base Model only:      LORA_PATH=\"\" ./serve_vllm_quantized_bnb_eessi.sh"
echo "================================="
echo

# =====================================================================
# STEP 1: Validation Sequence
# =====================================================================
echo "--- Validating model, tokenizer, and adapter paths ---"

if [[ ! -d "${ACTIVE_MODEL_DIR}" ]]; then
    echo "ERROR: BitsAndBytes quantized model not found at: ${ACTIVE_MODEL_DIR}"
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
# STEP 3: Build & Launch vLLM API Server
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

echo "--- Launching BitsAndBytes + LoRA vLLM Server ---"

vllm "${VLLM_ARGS[@]}"
