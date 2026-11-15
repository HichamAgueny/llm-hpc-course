#!/usr/bin/env bash

# Usage: ./gpu_monitor.sh <JOB_ID>
# Stop with Ctrl+C

set -u

JOB_ID="${1:-}"

if [[ -z "$JOB_ID" || ! "$JOB_ID" =~ ^[0-9]+$ ]]; then
    echo "Usage: $0 <JOB_ID>" >&2
    exit 1
fi

# Wait until the job starts.
while true; do
    JOB_STATE="$(squeue -h -j "$JOB_ID" -o "%T" 2>/dev/null | head -n 1)"

    if [[ -z "$JOB_STATE" ]]; then
        echo "Job $JOB_ID was not found. It may have finished, or it is not owned by this user." >&2
        exit 1
    fi

    case "$JOB_STATE" in
        PENDING)
            clear
            echo "================================================================================"
            echo " Job $JOB_ID is pending. Waiting for resources..."
            echo " Press Ctrl+C to exit."
            echo "================================================================================"
            sleep 5
            ;;

        RUNNING)
            break
            ;;

        COMPLETING)
            echo "Job $JOB_ID is completing. GPU monitoring is no longer available."
            exit 0
            ;;

        *)
            echo "Job $JOB_ID is in state $JOB_STATE. Cannot monitor GPUs." >&2
            exit 1
            ;;
    esac
done

# Resolve the allocated nodes.
NODELIST="$(
    scontrol show job "$JOB_ID" 2>/dev/null |
        awk -F'NodeList=' 'NF > 1 {print $2}' |
        awk '{print $1}' |
        grep -v '^(null)$' |
        head -n 1
)"

if [[ -z "$NODELIST" ]]; then
    echo "Unable to resolve the node list for job $JOB_ID." >&2
    exit 1
fi

mapfile -t NODES < <(
    scontrol show hostnames "$NODELIST" 2>/dev/null
)

if [[ "${#NODES[@]}" -eq 0 ]]; then
    echo "The job has no allocated nodes." >&2
    exit 1
fi

NODE_COUNT="${#NODES[@]}"

echo "Found $NODE_COUNT allocated node(s) for job $JOB_ID."
echo "Starting one persistent monitoring step..."
sleep 2

# This command runs once per allocated node.
# The monitoring loop is inside the command, so no new srun step
# is created during each refresh.
read -r -d '' MONITOR_COMMAND <<'EOF' || true
while true; do
    HOSTNAME="$(hostname -s)"
    CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:-all}"

    printf '\n[%s] CUDA_VISIBLE_DEVICES=%s | %s\n' \
        "$HOSTNAME" \
        "$CUDA_DEVICES" \
        "$(date '+%H:%M:%S')"

    nvidia-smi \
        --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw \
        --format=csv,noheader,nounits 2>/dev/null |
    awk -F', ' '
    {
        printf "  [GPU %s] %-20.20s | Util: %3s%% | Mem: %6s / %6s MiB | Temp: %3sC | Pwr: %3sW\n",
            $1, $2, $3, $4, $5, $6, $7
    }
    '

    sleep 2
done
EOF

# One persistent job step.
# --overlap allows this monitoring step to run alongside the training step.
srun \
    --jobid="$JOB_ID" \
    --job-name=gpu-monitor \
    --overlap \
    --nodes="$NODE_COUNT" \
    --ntasks="$NODE_COUNT" \
    --ntasks-per-node=1 \
    --cpus-per-task=1 \
    --exact \
    --label \
    bash -c "$MONITOR_COMMAND"

echo
echo "Job $JOB_ID is no longer running. GPU monitoring stopped."
