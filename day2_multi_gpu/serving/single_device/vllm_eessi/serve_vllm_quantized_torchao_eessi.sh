#!/bin/bash -e
#SBATCH --job-name=vllm-serve-single-torchao-eessi
#SBATCH --time=00:30:00
#SBATCH --partition=accel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem-per-gpu=96G
#SBATCH -o ./out/%x-%j.out
# Slurm account & reservation are set in course_env.sh - no need to add them here.
#
# Two ways to start this server:
#   sbatch serve_vllm_quantized_torchao_eessi.sh     -> batch job, log in ./out/
#   ./serve_vllm_quantized_torchao_eessi.sh          -> inside an interactive GPU shell (bash srun_sinteractive)
# Then, from a login node in this folder:  ./chat.sh

# --- Load the course settings (account, paths, containers, modules) ----------
for _d in "${SLURM_SUBMIT_DIR:-$PWD}" "$PWD" "${COURSE_ROOT:-/}"; do
    until [[ -f "$_d/course_env.sh" || "$_d" == "/" ]]; do _d="$(dirname "$_d")"; done
    if [[ -f "$_d/course_env.sh" ]]; then break; fi
done
if [[ ! -f "$_d/course_env.sh" ]]; then echo "ERROR: course_env.sh not found - submit from inside the course folder" >&2; exit 1; fi
source "$_d/course_env.sh"

export PYTHONNOUSERSITE=1

module load "${EESSI_MODULE}"
module load "${VLLM_MODULE}"
module load Transformers
module load torchao

export NUMEXPR_MAX_THREADS=72

export NUM_GPUS="${SLURM_GPUS_ON_NODE:-1}"
echo "--Node: $(hostname)"
echo "--Allocated GPUs: ${NUM_GPUS}"
echo

# =====================================================================
# --- SERVER SETTINGS (paths, model, adapter and container: see course_env.sh) ---
# =====================================================================
# Override at launch time, for example:
#   LORA_PATH="" ./serve_vllm_quantized_torchao_eessi.sh                       -> no LoRA adapter
#   LORA_PATH=/path/to/checkpoints/epoch_0 ./serve_vllm_quantized_torchao_eessi.sh
#   PORT=8001 ./serve_vllm_quantized_torchao_eessi.sh                          -> use another port
export BASE_MODEL_DIR="${MODEL_PATH:-${BASE_MODEL_DIR}}"
# Quantized model: your own copy from the quantization lab (results/quantized_models/)
# if it exists, otherwise the shared one. Override with QUANTIZED_MODEL_PATH=...
export QUANTIZED_MODEL_DIR="${QUANTIZED_MODEL_PATH:-$(course_quantized_model torchao_int8)}"

# Serve the TorchAO quantized model as the base weight layer
export ACTIVE_MODEL_DIR="${QUANTIZED_MODEL_DIR}"

# LoRA adapter (default from course_env.sh; LORA_PATH="" disables it)
export LORA_ADAPTER_DIR="${LORA_PATH-${LORA_ADAPTER_DIR}}"

export QUANTIZATION="${QUANTIZATION:-torchao}"
export MAX_LORA_RANK="${MAX_LORA_RANK:-64}"
export PORT="${PORT:-8000}"

# Connection file read by chat.sh (in this folder)
export CONNECTION_DIR="${COURSE_ROOT}/day2_multi_gpu/serving/single_device/vllm_eessi"
export CONNECTION_FILE="${CONNECTION_DIR}/connection.env"
mkdir -p "${CONNECTION_DIR}" "${HF_HOME}" "${VLLM_CACHE_ROOT}"
# =====================================================================


# -------- LoRA Detection --------
if [[ -n "${LORA_ADAPTER_DIR}" ]]; then
    USE_LORA=true
    MODEL_NAME="custom_lora"
else
    USE_LORA=false
    MODEL_NAME="${ACTIVE_MODEL_DIR}"
fi

echo "--- Course folder (host): ${COURSE_ROOT}"
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
echo "   • Run with LoRA with quantized base model(default):  ./serve_vllm_quantized_torchao_eessi.sh"
echo "   • Run Quantized Base Model only:      LORA_PATH=\"\" ./serve_vllm_quantized_torchao_eessi.sh"
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

echo "--- Launching TorchAO + LoRA vLLM Server ---"

vllm "${VLLM_ARGS[@]}"
