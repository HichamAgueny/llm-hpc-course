set -e

# Check JobID argument
if [ -z "$1" ]; then
    echo "Usage: $0 <JobID>"
    echo "Example: $0 12345678"
    exit 1
fi

JOB_ID="$1"

# Load Cray environment
module load CrayEnv
module load cray-python/3.12.12

# Prevent loading user-site packages
export PYTHONNOUSERSITE=1

export CLUSTER_PROJECT_ROOT="/cluster/work/projects/nn9971k"
export PROJECT_DIR="${CLUSTER_PROJECT_ROOT}/${USER}/llm-hpc-course"

# nvitop installation directory
NVITOP_DIR="${PROJECT_DIR}/apptainer/nvitop"

# Create installation directory
mkdir -p "$NVITOP_DIR"

# Configure environment
export PATH="${NVITOP_DIR}/bin:$PATH"
export PYTHONPATH="${NVITOP_DIR}:${PYTHONPATH:-}"

# Check whether nvitop executable exists
if [ ! -x "$NVITOP_DIR/bin/nvitop" ]; then
    echo "nvitop not found. Installing..."

    python3 -m pip install --upgrade --target="$NVITOP_DIR" nvitop

    echo "nvitop installation completed."
else
    echo "nvitop is already installed."
fi

# Verify installation
if [ ! -x "$NVITOP_DIR/bin/nvitop" ]; then
    echo "ERROR: nvitop executable not found."
    exit 1
fi

# Launch nvitop inside the existing Slurm allocation
echo "Launching nvitop on JobID: $JOB_ID"

exec srun --jobid="$JOB_ID" --overlap --pty nvitop
