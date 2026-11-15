# Utility Scripts

This directory contains utility scripts for monitoring and managing LLM fine-tuning and inference jobs on the HPC cluster.

## Scripts

### 1. GPU Monitor (`gpu.monitor.sh`)
Monitoring GPU utilization is crucial for optimizing your fine‑tuning, profiling, and inference jobs. This script provides real‑time feedback on GPU metrics (utilization, memory consumption, temperature) for a specific active Slurm job.

**Usage:**
```bash
./gpu.monitor.sh <JobID>
```

**Features:**
- Discovers the compute node allocated to `<JobID>` via `squeue`.
- Connects and monitors GPU device status in real time.
- Helps identify communication bottlenecks, idle cycles, and GPU memory saturation.

---

### 2. Interactive Monitoring with nvitop (`run_nvitop.sh`)
Launches the interactive curses-based `nvitop` GPU process monitor. See [install_nvitop.md](./install_nvitop.md) for local installation instructions.

**Usage:**
```bash
./run_nvitop.sh <JobID>
```

---

## Integration with Course Labs
Across the labs in `day1_single_gpu/` and `day2_multi_gpu/`, you are encouraged to run `gpu.monitor.sh` in a separate terminal window or pane once your Slurm job starts running.
