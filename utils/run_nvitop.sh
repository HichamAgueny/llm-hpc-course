#!/bin/bash
set -e

# Check JobID argument
if [ -z "$1" ]; then
    echo "Usage: $0 <JobID>"
    echo "Example: $0 12345678"
    exit 1
fi

JOB_ID="$1"

# Prevent loading user-site packages
export PYTHONNOUSERSITE=1

export https_proxy=http://10.63.2.48:3128/
export http_proxy=http://10.63.2.48:3128/

export CLUSTER_PROJECT_ROOT="/cluster/work/projects/nn9971k"
export PROJECT_DIR="${CLUSTER_PROJECT_ROOT}/${USER}/llm-hpc-course"
UTILS_DIR="${PROJECT_DIR}/utils"

# Virtual environment directory for nvitop (inside utils)
VENV_DIR="${UTILS_DIR}/nvitop_env"

# Identify the compute node from the JobID
JOB_STATE=$(squeue -j "$JOB_ID" -h -o "%T" 2>/dev/null || true)
if [ "$JOB_STATE" != "RUNNING" ]; then
    echo "ERROR: Job $JOB_ID is not running (state: ${JOB_STATE:-not found})."
    exit 1
fi

NODELIST=$(squeue -j "$JOB_ID" -h -o "%N")
NODE=$(scontrol show hostnames "$NODELIST" | head -n 1)
echo "Job $JOB_ID is running on node: $NODE"

# SSH to the compute node, cd to utils, and install nvitop there if needed
ssh "$NODE" bash -l <<EOF
set -e
module load CrayEnv
module load cray-python/3.12.12
export PYTHONNOUSERSITE=1
export https_proxy=${https_proxy}
export http_proxy=${http_proxy}

cd "${UTILS_DIR}"
echo "On \$(hostname), in \$(pwd)"

if [ ! -f "${VENV_DIR}/bin/nvitop" ]; then
    echo "nvitop virtual environment not found. Creating and installing..."
    python3 -m venv "${VENV_DIR}"
    source "${VENV_DIR}/bin/activate"
    pip install nvitop
    echo "nvitop installation completed."
else
    echo "nvitop virtual environment already exists."
fi
EOF

# Verify installation
if [ ! -x "$VENV_DIR/bin/nvitop" ]; then
    echo "ERROR: nvitop executable not found."
    exit 1
fi

# Launch nvitop inside the existing Slurm allocation
echo "Launching nvitop on JobID: $JOB_ID"

exec srun --jobid="$JOB_ID" --overlap --pty "$VENV_DIR/bin/nvitop"
