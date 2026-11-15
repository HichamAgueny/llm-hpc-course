#!/bin/bash
# Terminal chat client for the vLLM server started from this folder.
# Run it on a login node:  ./chat.sh
# (the server writes connection.env next to this script when it starts)

CONNECTION_FILE="${CONNECTION_FILE:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/connection.env}"

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
