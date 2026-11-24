#!/bin/bash -e
#SBATCH --job-name=vllm-multinodes
#SBATCH --account=nn9997k
#SBATCH --time=00:15:00
#SBATCH --partition=accel
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=288
#SBATCH -o ./out/%x-%j.out
#SBATCH --mem-per-gpu=96G
#SBATCH --reservation=software
#SBATCH --exclusive

set -o pipefail

echo "--Node: $(hostname)"
echo

export PYTHONNOUSERSITE=1

# Load cluster high-performance communication environment components

module load NRIS/GPU
module load NCCL/2.30.4-GCCcore-14.3.0-CUDA-13.0.0

export NUMEXPR_MAX_THREADS=72

# =====================================================================
# --- CONFIGURABLE VARIABLES AND PATHS (USERS ONLY MODIFY THIS BLOCK) ---
# =====================================================================
# 1. Base Project Allocation
export CLUSTER_PROJECT_ROOT="/cluster/work/projects/nn9971k"

# 2. Personal Workspace Setup
export PROJECT_DIR="${CLUSTER_PROJECT_ROOT}/${USER}/llm-hpc-course"
export SHARED_DIR="${PROJECT_DIR}/shared"
export RESULT_DIR="${PROJECT_DIR}/results/checkpoint_results"
export CURRENT_DIR="${PROJECT_DIR}/day2_multi_gpu/inference"

# 3. Model, Adapter & Serving Parameters
export BASE_MODEL_DIR="${MODEL_PATH:-${SHARED_DIR}/models/Llama-3.2-1B-Instruct}"
export LORA_ADAPTER_DIR="${LORA_PATH:-${RESULT_DIR}/llama3.2-1b_lora_single_device/checkpoints/epoch_1}"
export PROMPT_FILE="${PROMPT_FILE:-${CURRENT_DIR}/prompt_QA.json}"
export QUANTIZATION="${QUANTIZATION:-None}"
export MAX_LORA_RANK="${MAX_LORA_RANK:-64}"

# 4. Apptainer Container Configuration
export CONTAINER_DIR="${PROJECT_DIR}/apptainer"
export APPTAINER_SIF="${CONTAINER_DIR}/vllm0.12_cu131_py3.12_arm_custom.sif"

# 5. Internal Multi-Node Network Helper Path
export MyCurrentWD="${PROJECT_DIR}/day2_multi_gpu/inference"

# 6. Script & Parallelism Settings
export PYTHON_FILE="${PROJECT_DIR}/recipes/inference/vllm_distributed_ray.py"

# 7. Caches under project share
export HF_HOME="${PROJECT_DIR}/hf_cache"
export VLLM_CACHE_ROOT="${PROJECT_DIR}/.cache/vllm"
mkdir -p "${HF_HOME}" "${VLLM_CACHE_ROOT}"


# =====================================================================
# --- System & High-Speed Network Interconnect Paths ---
# =====================================================================

export HOST_NCCL_PATH="${NCCL_HOME:-/cluster/software/NRIS/neoverse_v2/software/NCCL/2.30.4-GCCcore-14.3.0-CUDA-13.0.0}"
export HOST_AWS_OFI_NCCL_PATH="${EBROOTAWSMINOFIMINNCCL:-/cluster/software/NRIS/neoverse_v2/software/aws-ofi-nccl/1.19.1-GCCcore-14.3.0-CUDA-13.0.0}/lib"
export HOST_LIBFABRIC_LIB_PATH="/opt/cray/libfabric/2.3.1/lib64"
export HOST_LIBFABRIC_INCLUDE_PATH="/opt/cray/libfabric/2.3.1/include"
export HOST_CXI_LIB_PATH="/usr/lib64"

# =====================================================================
# --- Slurm Settings ---
# =====================================================================
N=$SLURM_JOB_NUM_NODES
nproc_perN=$SLURM_NTASKS_PER_NODE
total_nbr_gpus=$((N * SLURM_GPUS_ON_NODE))

# Tensor parallelism within each node
export TP_SIZE=$SLURM_GPUS_ON_NODE
# Pipeline parallelism across nodes
export PP_SIZE=$SLURM_JOB_NUM_NODES

echo "SLURM Job ID: $SLURM_JOB_ID"
echo "--nbr of nodes: $N"
echo "--nbr of GPUs per node: $SLURM_GPUS_ON_NODE"
echo "--Total nbr of GPUs: $total_nbr_gpus"
echo "--Tensor parallel size: $TP_SIZE"
echo "--Pipeline parallel size: $PP_SIZE"
echo

# --- Locale Settings ---
export LANG=en_US.UTF-8
export LC_ALL=en_US.UTF-8

# =====================================================================
# --- Verify Required Files ---
# =====================================================================
if [[ ! -f "$PROMPT_FILE" ]]; then
    echo "ERROR: Prompt file not found: $PROMPT_FILE"
    exit 1
fi

if [[ ! -f "$APPTAINER_SIF" ]]; then
    echo "ERROR: Container not found: $APPTAINER_SIF"
    exit 1
fi

if [[ ! -f "$PYTHON_FILE" ]]; then
    echo "ERROR: Python script not found: $PYTHON_FILE"
    exit 1
fi

if [[ ! -f "${MyCurrentWD}/nccl_env.sh" ]]; then
    echo "ERROR: ${MyCurrentWD}/nccl_env.sh not found!"
    exit 1
fi


# =====================================================================
# --- Getting the Node Names ---
# =====================================================================
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)

export head_node=${nodes_array[0]}

# Resolve the IPv4 address of the head node
export head_node_ip=$(getent ahostsv4 "$head_node" | awk 'NR==1 {print $1}')

if [[ -z "$head_node_ip" ]]; then
    echo "ERROR: Could not resolve IP address of $head_node"
    exit 1
fi

echo "Head node: $head_node"
echo "Head node IP: $head_node_ip"
echo

# =====================================================================
# --- vLLM, Ray and NCCL Environment Settings ---
# =====================================================================
export APPTAINERENV_VLLM_LOGGING_LEVEL=INFO
export APPTAINERENV_VLLM_WORKER_MULTIPROC_METHOD=spawn

# Workaround for AOT compilation with pipeline parallelism
export APPTAINERENV_VLLM_USE_AOT_COMPILE=0

# Communication debugging
export APPTAINERENV_NCCL_DEBUG=INFO
export APPTAINERENV_NCCL_DEBUG_SUBSYS=INIT,NET,GRAPH,ENV
export APPTAINERENV_RAY_DEDUP_LOGS=0
export APPTAINERENV_PYTHONUNBUFFERED=1

# Explicitly select host NCCL library
export APPTAINERENV_VLLM_NCCL_SO_PATH=/opt/nccl/lib/libnccl.so.2

# =====================================================================
# --- Define the Inference Task Function ---
# =====================================================================
run_inference() {

    python "${PYTHON_FILE}" \
        --model "$BASE_MODEL_DIR" \
        --lora-path "$LORA_ADAPTER_DIR" \
        --prompt-file "$PROMPT_FILE" \
        --quantization "$QUANTIZATION" \
        --max-lora-rank "$MAX_LORA_RANK" \
        --tensor-parallel-size "$TP_SIZE" \
        --pipeline-parallel-size "$PP_SIZE"
}

# =====================================================================
# --- Configuration Summary ---
# =====================================================================
echo "----------------------------------------"
echo "Configuration:"
echo "  Base Model: $BASE_MODEL_DIR"
echo "  LoRA Adapter: $LORA_ADAPTER_DIR"
echo "  Prompt file: $PROMPT_FILE"
echo "  Quantization: $QUANTIZATION"
echo "  Max LoRA Rank: $MAX_LORA_RANK"
echo "  Tensor Parallel Size: $TP_SIZE"
echo "  Pipeline Parallel Size: $PP_SIZE"
echo "----------------------------------------"

# =====================================================================
# --- Ray Cluster Configuration ---
# =====================================================================
export port=6379
export ip_head="${head_node_ip}:${port}"
export RAY_ADDRESS="$ip_head"

echo "IP Head: $ip_head"
# Shared directory for coordinating Ray startup and shutdown
export RAY_SYNC_DIR=$(mktemp -d \
    "${PROJECT_DIR}/.ray-${SLURM_JOB_ID}.XXXXXX")

# Remove synchronization directory when the job exits
trap 'rm -rf -- "$RAY_SYNC_DIR"' EXIT

# =====================================================================
# --- Define Ray Execution Function ---
# =====================================================================

run_ray_node() {

    set -o pipefail

    # -------------------------------------------------------------
    # Source local cluster network environment inside Apptainer

    if [[ -f "${MyCurrentWD}/nccl_env.sh" ]]; then
        source "${MyCurrentWD}/nccl_env.sh"
    else
        echo "ERROR: ${MyCurrentWD}/nccl_env.sh not found!"
        exit 1
    fi

    # Configure communication library search paths
    export LD_LIBRARY_PATH="/opt/nccl/lib:/opt/aws-ofi-nccl/lib:/opt/libfabric/lib:/opt/cxi:${LD_LIBRARY_PATH:-}"

    # Explicitly select the host AWS OFI NCCL network plugin
    export NCCL_NET_PLUGIN=/opt/aws-ofi-nccl/lib/libnccl-net.so

    # Verify that the required shared libraries exist
    if [[ ! -r "$VLLM_NCCL_SO_PATH" ]]; then
        echo "ERROR: NCCL library not found: $VLLM_NCCL_SO_PATH"
        exit 1
    fi

    if [[ ! -r "$NCCL_NET_PLUGIN" ]]; then
        echo "ERROR: NCCL network plugin not found: $NCCL_NET_PLUGIN"
        exit 1
    fi

    # -------------------------------------------------------------
    # Configure the local node IP
    # -------------------------------------------------------------

    node_ip=$(getent ahostsv4 \
        "${SLURMD_NODENAME}" | awk 'NR==1 {print $1}')

    if [[ -z "$node_ip" ]]; then
        echo "ERROR: Could not resolve IP of ${SLURMD_NODENAME}"
        exit 1
    fi

    export VLLM_HOST_IP="$node_ip"

    # -------------------------------------------------------------
    # Display communication environment
    # -------------------------------------------------------------
    echo "----------------------------------------"
    echo "Node: $(hostname)"
    echo "Node IP: $node_ip"
    echo "SLURM_STEP_ID: ${SLURM_STEP_ID}"
    echo "SLURM_PROCID: ${SLURM_PROCID}"
    echo "SLINGSHOT_VNIS: ${SLINGSHOT_VNIS:-NOT SET}"
    echo "SLINGSHOT_DEVICES: ${SLINGSHOT_DEVICES:-NOT SET}"
    echo "SLINGSHOT_SVC_IDS: ${SLINGSHOT_SVC_IDS:-NOT SET}"
    echo "VLLM_NCCL_SO_PATH: ${VLLM_NCCL_SO_PATH}"
    echo "NCCL_NET_PLUGIN: ${NCCL_NET_PLUGIN}"
    echo "NCCL_NET: ${NCCL_NET:-NOT SET}"
    echo "NCCL_NET_GDR_LEVEL: ${NCCL_NET_GDR_LEVEL:-NOT SET}"
    echo "LD_LIBRARY_PATH: ${LD_LIBRARY_PATH}"
    echo "----------------------------------------"

    # =============================================================
    # --- Starting the Ray Head Node ---
    # =============================================================

    if [[ "${SLURMD_NODENAME}" == "$head_node" ]]; then
        echo "Starting HEAD at $(hostname)"

        export VLLM_HOST_IP="$head_node_ip"

        # Signal completion and stop Ray when the head process exits
        trap 'touch "$RAY_SYNC_DIR/done"; ray stop || true' EXIT

        ray start --head \
            --node-ip-address="$head_node_ip" \
            --port="$port" \
            --num-cpus "${SLURM_CPUS_PER_TASK}" \
            --num-gpus "${SLURM_GPUS_ON_NODE}"

        # Signal that the Ray head has started
        touch "${RAY_SYNC_DIR}/head_ready"

        # ---------------------------------------------------------
        # Wait for all Ray nodes and GPUs to become available
        # ---------------------------------------------------------

        echo "Waiting for all Ray nodes to join the cluster"

        timeout 120s python - <<'PY'
import os
import time
import ray

ray.init(address=os.environ["RAY_ADDRESS"])

expected_nodes = int(os.environ["SLURM_JOB_NUM_NODES"])

expected_gpus = (
    expected_nodes * int(os.environ["SLURM_GPUS_ON_NODE"])
)

while True:

    nodes = sum(node["Alive"] for node in ray.nodes())

    gpus = ray.cluster_resources().get("GPU", 0)

    print(
        f"Ray resources: {nodes}/{expected_nodes} nodes, "
        f"{gpus}/{expected_gpus} GPUs",
        flush=True,
    )

    if nodes >= expected_nodes and gpus >= expected_gpus:
        break

    time.sleep(2)

ray.shutdown()
PY

        echo
        echo "Ray cluster is ready"
        ray status

        # ---------------------------------------------------------
        # Execute inference ONLY on the Ray head node
        # ---------------------------------------------------------
        echo
        echo "Starting distributed inference"
        echo

        run_inference

    # =============================================================
    # --- Starting the Ray Worker Nodes ---
    # =============================================================
    else
        echo "Starting WORKER at $(hostname)"

        # Wait until the head node is ready

        start_time=$SECONDS

        while [[ ! -f "${RAY_SYNC_DIR}/head_ready" ]]; do
            if [[ -f "${RAY_SYNC_DIR}/done" ]]; then
                echo "ERROR: Ray head exited before initialization"
                exit 1
            fi
            if (( SECONDS - start_time > 120 )); then
                echo "ERROR: Ray head startup timed out"
                exit 1
            fi
            sleep 2
        done

        # Stop the local Ray runtime when this task exits
        trap 'ray stop || true' EXIT

        # Connect the worker node to the Ray cluster
        ray start --address "$ip_head" \
            --node-ip-address="$node_ip" \
            --num-cpus "${SLURM_CPUS_PER_TASK}" \
            --num-gpus "${SLURM_GPUS_ON_NODE}"

        # Keep the worker alive until inference finishes
        while [[ ! -f "${RAY_SYNC_DIR}/done" ]]; do
            sleep 2
        done

        echo "Inference completed. Stopping worker at $(hostname)"
    fi
}

# =====================================================================
# --- Execute Ray and vLLM Using ONE Multi-Node Slurm Step ---
# =====================================================================

time srun \
    --network=disable_rdzv_get \
    --nodes="${SLURM_JOB_NUM_NODES}" \
    --ntasks="${SLURM_JOB_NUM_NODES}" \
    --ntasks-per-node=1 \
    --cpu-bind=cores \
    --kill-on-bad-exit=1 \
    apptainer exec --nv \
    --bind "${PROJECT_DIR}:/workspace" \
    --bind "${CLUSTER_PROJECT_ROOT}" \
    --bind /cluster/software \
    --bind "${HOST_NCCL_PATH}:/opt/nccl" \
    --bind "${HOST_AWS_OFI_NCCL_PATH}:/opt/aws-ofi-nccl/lib" \
    --bind "${HOST_LIBFABRIC_LIB_PATH}:/opt/libfabric/lib" \
    --bind "${HOST_LIBFABRIC_INCLUDE_PATH}:/opt/libfabric/include" \
    --bind "${HOST_CXI_LIB_PATH}:/opt/cxi" \
    "${APPTAINER_SIF}" \
    bash -ec "$(declare -f run_inference); $(declare -f run_ray_node); run_ray_node"

echo

# =====================================================================
# --- Extract and Display Inference Results ---
# =====================================================================

LOG_FILE="${SLURM_SUBMIT_DIR}/out/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out"

echo
echo "============================================================"
echo "                  INFERENCE RESULTS"
echo "============================================================"
echo

awk '
    /^ALL INFERENCE COMPLETED SUCCESSFULLY/ { exit }
    /^\[(Base|LoRA)\] Prompt [0-9]+:/ { show=1 }
    /^[-~]+$/ && length($0)>=60 { show=0 }
    show { print }
' "$LOG_FILE"

echo "============================================================"
echo "                  END OF INFERENCE RESULTS"
echo "============================================================"
echo "--- Finished :) ---"

echo "Job finished at: $(date)"
