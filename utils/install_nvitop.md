1. allocate an interactive node

2. load python e.g.
module load CrayEnv
module load cray-python/3.12.12

export PYTHONNOUSERSITE=1

mkdir -p /cluster/projects/nn9997k/$USER/llm-hpc-course/apptainer/nvitop

python3 -m pip install --target=/cluster/projects/nn9997k/$USER/llm-hpc-course/apptainer/nvitop nvitop

export PATH=/cluster/projects/nn9997k/$USER/llm-hpc-course/apptainer/nvitop/bin:$PATH
export PYTHONPATH=/cluster/projects/nn9997k/$USER/llm-hpc-course/apptainer/nvitop:$PYTHONPATH

nvitop
