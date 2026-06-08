#!/bin/bash

PROJECT_DIR="/cluster/work/projects/nn9970k"
MyWD="$PROJECT_DIR/$USER/llm-hpc-course"
CONNECTION_FILE="${MyWD}/day2_multi_gpu/serving/single_device/connection.env"

# Wait until the server connection file exists
echo "Waiting for server to start..."
while [[ ! -f "$CONNECTION_FILE" ]]; do
    sleep 5
done

# Load host, port and model from the server script
source "$CONNECTION_FILE"

echo "Connected to $HOST:$PORT"
echo "Model: $MODEL"
echo "Type 'exit' or 'quit' to end the conversation."
echo "----------------------------------------"

while true; do
    echo -n "You: "
    read -r prompt

    # Graceful exit
    if [[ "$prompt" == "exit" || "$prompt" == "quit" ]]; then
        echo "Exiting chat session."
        break
    fi

    # Skip empty lines
    if [[ -z "$prompt" ]]; then
        continue
    fi

    echo -n "A: "

    jq -n \
      --arg model "$MODEL" \
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
