#!/bin/bash

# =====================================================================
# --- CONFIGURABLE VARIABLES AND PATHS ---
# =====================================================================
export CLUSTER_PROJECT_ROOT="/cluster/projects/nn9997k"
export PROJECT_DIR="${CLUSTER_PROJECT_ROOT}/${USER}/llm-hpc-course"
export CONNECTION_FILE="${PROJECT_DIR}/day2_multi_gpu/serving/single_device/vllm_eessi/connection.env"
# =====================================================================

# Wait until the server connection file exists
echo "Waiting for server connection file at: ${CONNECTION_FILE}..."
while [[ ! -f "${CONNECTION_FILE}" ]]; do
    sleep 5
done

# Load HOST, PORT, and MODEL from the server script
source "${CONNECTION_FILE}"

echo "Connected to ${HOST}:${PORT} (Model: ${MODEL})"
echo "Type 'exit' or 'quit' to end the conversation."
echo "----------------------------------------"

while true; do
    # Read user input safely
    echo -n "You: "
    read -r prompt

    # Graceful exit condition
    if [[ "${prompt}" == "exit" || "${prompt}" == "quit" ]]; then
        echo "Exiting chat session."
        break
    fi

    # Skip empty lines
    if [[ -z "${prompt}" ]]; then
        continue
    fi

    echo -n "A: "

    # 1. Dynamically construct JSON payload using $MODEL from connection.env
    # 2. POST to the OpenAI-compatible endpoint
    # 3. Parse out the message content cleanly
    jq -n \
      --arg model "${MODEL}" \
      --arg prompt "${prompt}" \
      '{
        model: $model,
        messages: [{role: "user", content: $prompt}],
        max_tokens: 512
      }' | curl -s -X POST "${HOST}:${PORT}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d @- | jq -r '.choices[0].message.content // .error.message // "Error: Empty response or invalid payload received."'

    echo -e "\n---"
done
