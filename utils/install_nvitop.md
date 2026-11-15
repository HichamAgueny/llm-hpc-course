# Installing nvitop

The easiest way: `run_nvitop.sh <JobID>` installs nvitop automatically on first use and attaches
it to your running job:
```bash
source course_env.sh          # once per login (puts utils/ on your PATH)
run_nvitop.sh <JobID>
```

## Manual installation (optional)

1. Allocate an interactive node (e.g. `bash apptainer/srun_interactive`).
2. Install nvitop into the course folder:
```bash
source course_env.sh
module load CrayEnv
module load cray-python/3.12.12
export PYTHONNOUSERSITE=1

mkdir -p $COURSE_ROOT/apptainer/nvitop
python3 -m pip install --target=$COURSE_ROOT/apptainer/nvitop nvitop

export PATH=$COURSE_ROOT/apptainer/nvitop/bin:$PATH
export PYTHONPATH=$COURSE_ROOT/apptainer/nvitop:$PYTHONPATH

nvitop
```
