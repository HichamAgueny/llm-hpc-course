#!/bin/bash

PROJECT_DIR="/cluster/projects/nn9997k"
MyWD="$PROJECT_DIR/$USER/llm-hpc-course"
CONNECTION_FILE="${MyWD}/day2_multi_gpu/serving/single_device/connection.env"

# Wait until the server connection file exists
echo "Waiting for server to start..."
while [[ ! -f "$CONNECTION_FILE" ]]; do
    sleep 5
done

# Load host and port from the server script
source "$CONNECTION_FILE"

echo "Connected to $HOST:$PORT"
echo "Type 'exit' or 'quit' to end the conversation."
echo "----------------------------------------"

while true; do
    # Read user input safely
    echo -n "You: "
    read -r prompt

    # Graceful exit condition
    if [[ "$prompt" == "exit" || "$prompt" == "quit" ]]; then
        echo "Exiting chat session."
        break
    fi

    # Skip empty lines
    if [[ -z "$prompt" ]]; then
        continue
    fi

    echo -n "A: "

    # 1. Dynamically construct a safe JSON payload using jq
    # 2. POST to the OpenAI-compatible endpoint
    # 3. Parse out the message content cleanly
    jq -n \
      --arg model "custom_lora" \
      --arg prompt "$prompt" \
      '{
        model: $model,
        messages: [{role: "user", content: $prompt}],
        max_tokens: 512
      }' | curl -s -X POST "$HOST:$PORT/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d @- | jq -r '.choices[0].message.content // "Error: Empty response or invalid payload received."'

    echo -e "\n---"
done
