#!/bin/bash -e
#SBATCH --job-name=qat-llama3-1B-1gpu
#SBATCH --reservation=LLM_in_person_course
#SBATCH --account=nn9970k
#SBATCH --time=00:10:00
#SBATCH --partition=accel
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH -o ./out/%x-%j.out
#SBATCH --mem-per-gpu=97G

echo "--Node: $(hostname)"
echo

# --- Variables and Paths (HOST-SIDE) ---
PROJECT_DIR="/cluster/work/projects/nn9970k"
MyWD="$PROJECT_DIR/$USER/llm-hpc-course"
CONTAINER_DIR="${MyWD}/apptainer"
APPTAINER_SIF="${CONTAINER_DIR}/vllm0.12_cu131_py3.12_arm_custom.sif"


# Configs and python files for fine-tuning
PYTHON_FILE="${MyWD}/recipes/quantization/qat_torchoa.py"

# Define paths for QAT with torchAO
MODEL_DIR="${MyWD}/shared/models/Llama-3.2-1B-Instruct"
OUT_DIR="${MyWD}/shared/models/Llama-3.2-1B-Instruct-torchao_1GPU"

# Check if output directory exists; if not, create it
if [ ! -d "$OUT_DIR" ]; then
    echo "Output directory does not exist. Creating: $OUT_DIR"
    mkdir -p "$OUT_DIR"
fi

echo "--- My Main Directory (host): ${MyWD}"
echo "--- Bind-mounted inside container as: /workspace"
echo

echo "=== Running inside Apptainer ==="
echo "PYTHON_FILE: ${PYTHON_FILE}"
echo

# --- Slurm setting
N=$SLURM_JOB_NUM_NODES
nproc_perN=$SLURM_NTASKS_PER_NODE
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "--nbr of nodes: $N"
echo "--nbr of GPUs: $nproc_perN"
echo
INNER_SCRIPT_TEMP="./.my_script_temp_${SLURM_JOB_ID}"
cat > "${INNER_SCRIPT_TEMP}" << EOF
#!/bin/bash -e
export PYTHONNOUSERSITE=1
# Create virtual env only if it doesn't already exist
if [ ! -d "$CONTAINER_DIR/MyEn" ]; then
    echo "Creating virtual environment at $CONTAINER_DIR/MyEn..."
    python -m venv "$CONTAINER_DIR/MyEn" --system-site-packages
else
    echo "Virtual environment already exists."
fi

# Activate the env. var.
source $CONTAINER_DIR/MyEn/bin/activate

# Check if 'accelerate' is installed. If not, install it.
if ! pip show accelerate > /dev/null 2>&1; then
    echo "Package 'accelerate' not found. Installing..."
    pip install accelerate
else
    echo "Package 'accelerate' is already installed. Skipping installation."
fi

# Run the qat with torchao script
python "${PYTHON_FILE}" --model_path "$MODEL_DIR" --output_path "$OUT_DIR"

EOF

chmod +x "${INNER_SCRIPT_TEMP}"

echo "--- Launching the application inside Apptainer ---"

# --- Execute with Apptainer ---
# Bind host project directory to /workspace inside container
# --nv enables NVIDIA GPU support

time srun apptainer exec --nv \
      -B "${MyWD}:/workspace" \
      -B $PROJECT_DIR \
      "${APPTAINER_SIF}" \
      ${INNER_SCRIPT_TEMP}

echo
echo "--- Finished :) ---"
