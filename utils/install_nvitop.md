# Installing nvitop

The easiest way: `utils/run_nvitop.sh <JobID>` installs nvitop on first use and attaches it to your job.

Manual installation:

1. allocate an interactive node

2. load python e.g.
module load CrayEnv
module load cray-python/3.12.12

export PYTHONNOUSERSITE=1

mkdir -p /cluster/work/projects/nn9971k/$USER/llm-hpc-course/apptainer/nvitop

python3 -m pip install --target=/cluster/work/projects/nn9971k/$USER/llm-hpc-course/apptainer/nvitop nvitop

export PATH=/cluster/work/projects/nn9971k/$USER/llm-hpc-course/apptainer/nvitop/bin:$PATH
export PYTHONPATH=/cluster/work/projects/nn9971k/$USER/llm-hpc-course/apptainer/nvitop:$PYTHONPATH

nvitop
