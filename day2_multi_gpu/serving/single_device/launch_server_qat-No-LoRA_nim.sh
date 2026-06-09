#!/bin/bash -e

export NUM_GPUS=$SLURM_GPUS_ON_NODE
echo "--nbr of GPUs: $NUM_GPUS"
echo

# -------- User configuration --------
PROJECT_DIR="/cluster/work/projects/nn9970k"
MyWD="$PROJECT_DIR/$USER/llm-hpc-course"
APPTAINER_SIF="${MyWD}/apptainer/vllm0.12_cu131_py3.12_arm_custom.sif"

# Set paths
export MODEL_PATH=${MODEL_PATH:-"${MyWD}/shared/models/Llama-3.2-1B-Instruct"}
export QUANTIZED_MODEL_PATH=${QUANTIZED_MODEL_PATH:-"${MyWD}/shared/models/Llama-3.2-1B-Instruct-torchao_1GPU"}
export LORA_PATH=${LORA_PATH:-""}                # Leave empty to disable LoRA
export QUANTIZATION=${QUANTIZATION:-"torchao"}
export MAX_LORA_RANK=${MAX_LORA_RANK:-64}


# -------- LoRA detection --------
if [[ -n "$LORA_PATH" ]]; then
    USE_LORA=true
    MODEL_NAME="custom_lora"
else
    USE_LORA=false
    MODEL_NAME="$MODEL_PATH"
fi

echo "----------------------------------------"
echo "Configuration:"
echo "  Base model:   $MODEL_PATH"
echo "  Quant model:  $QUANTIZED_MODEL_PATH"
echo "  LoRA:         ${LORA_PATH:-disabled}"
echo "  Quantization: $QUANTIZATION"
echo "  GPUs:         $NUM_GPUS"
echo "  Status:       Starting API Server on Port 8000..."
echo "----------------------------------------"

# -------- Validation --------
if [[ ! -d "$MODEL_PATH" ]]; then
    echo "ERROR: Base model not found: $MODEL_PATH"
    exit 1
fi

if [[ "$USE_LORA" == true && ! -d "$LORA_PATH" ]]; then
    echo "ERROR: LoRA checkpoint not found: $LORA_PATH"
    exit 1
fi

if [[ "$USE_LORA" == true && ! -f "$LORA_PATH/adapter_config.json" ]]; then
    echo "ERROR: adapter_config.json not found in $LORA_PATH"
    exit 1
fi

# -------- Cache --------
export VLLM_CACHE_ROOT=$MyWD/.cache/vllm
mkdir -p "$VLLM_CACHE_ROOT"
#export VLLM_LOGGING_LEVEL=ERROR

#Uncomment to avoid error "Ran out of input" when loading AOT compilation
#export VLLM_USE_AOT_COMPILE=0

# -------- Write connection info for chat.sh --------
CONNECTION_FILE="${MyWD}/day2_multi_gpu/serving/single_device/connection.env"
echo "HOST=http://$(hostname)" > "$CONNECTION_FILE"
echo "PORT=8000"               >> "$CONNECTION_FILE"
echo "MODEL=$QUANTIZED_MODEL_PATH"       >> "$CONNECTION_FILE"
echo "  Connection info written to: $CONNECTION_FILE"
echo "  Host: http://$(hostname):8000"
echo "  Model name for chat.sh: QUANTIZED_MODEL_PATH"

# -------- Build vLLM launch command --------
# https://docs.vllm.ai/en/latest/features/quantization/
#https://docs.vllm.ai/en/v0.12.0/features/quantization/torchao/
VLLM_CMD="vllm serve $QUANTIZED_MODEL_PATH
    --tensor-parallel-size $NUM_GPUS
    --host 0.0.0.0
    --port 8000
    --quantization $QUANTIZATION"

if [[ "$USE_LORA" == true ]]; then
    VLLM_CMD="$VLLM_CMD
    --enable-lora
    --lora-modules custom_lora=$LORA_PATH
    --max-lora-rank $MAX_LORA_RANK"
fi

# -------- Launch API Server --------
echo "Launching vLLM server (LoRA: $USE_LORA)..."
export MOUNT_DIR="$PROJECT_DIR/$USER"
export WORKDIR="/workspace"
apptainer exec --nv \
     -B ${MOUNT_DIR}:${WORKDIR} \
     -B $PROJECT_DIR \
     --pwd ${WORKDIR} \
     "${APPTAINER_SIF}" \
      $VLLM_CMD

